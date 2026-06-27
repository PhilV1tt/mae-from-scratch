# Donnees STL-10 / NEU, pretraining MAE, linear probe, reconstruction.
# Backend torch (cf mae.py) : les images sont placees sur le device (GPU MPS) avant la boucle.

import numpy as np
import torch
import mae as M

DATA = "data/stl10_binary"


def load_stl(split, n=None, offset=0):
    xf = {"unlabeled": "unlabeled_X.bin", "train": "train_X.bin", "test": "test_X.bin"}[split]
    px = 96 * 96 * 3
    with open(f"{DATA}/{xf}", "rb") as f:
        f.seek(offset * px)
        buf = np.fromfile(f, dtype=np.uint8, count=(-1 if n is None else n * px))
    imgs = buf.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1).astype(np.float32) / 255.0
    labels = None
    if split in ("train", "test"):
        yf = {"train": "train_y.bin", "test": "test_y.bin"}[split]
        labels = np.fromfile(f"{DATA}/{yf}", dtype=np.uint8).astype(np.int64) - 1
        if n is not None:
            labels = labels[offset:offset + n]
    return imgs, labels


def load_neu(img_size=96, seed=0, test_frac=0.2):
    import os, re, glob
    import imageio.v2 as iio
    from scipy.ndimage import zoom
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    cidx = {c: i for i, c in enumerate(classes)}
    files = sorted(glob.glob("data/neu/train/train/images/*.jpg")
                   + glob.glob("data/neu/valid/valid/images/*.jpg"))
    X, y = [], []
    for f in files:
        cls = re.sub(r"_\d+\.jpg$", "", os.path.basename(f))
        im = iio.imread(f)
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        z = img_size / im.shape[0]
        im = zoom(im, (z, img_size / im.shape[1], 1), order=1)[:img_size, :img_size]
        X.append(im.astype(np.float32) / 255.0)
        y.append(cidx[cls])
    X, y = np.array(X), np.array(y)
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in range(len(classes)):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        nte = int(len(idx) * test_frac)
        te += list(idx[:nte]); tr += list(idx[nte:])
    tr, te = np.array(tr), np.array(te); rng.shuffle(tr); rng.shuffle(te)
    return X[tr], y[tr], X[te], y[te], classes


def lr_schedule(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + np.cos(np.pi * t))


def pretrain(cfg, imgs, epochs=40, batch=256, base_lr=8e-4, wd=0.05, b2=0.95,
             mask_ratio=None, seed=0, log=True):
    torch.manual_seed(seed)
    model = M.MAE(cfg)
    opt = M.Adam(model.params(), lr=base_lr, b2=b2, wd=wd)
    X = M._t(imgs)                          # tout le jeu sur le device
    n = len(imgs); spe = n // batch; total = epochs * spe; warmup = max(1, int(0.08 * total))
    hist = []; step = 0
    for ep in range(epochs):
        order = torch.randperm(n, device=X.device)
        ep_loss = torch.zeros((), device=X.device)
        for it in range(spe):
            xb = X[order[it * batch:(it + 1) * batch]]
            opt.lr = lr_schedule(step, total, base_lr, warmup)
            loss, _, _, _ = model.forward(xb, mask_ratio)
            model.backward(); opt.step()
            ep_loss = ep_loss + loss.detach(); step += 1
        hist.append((ep_loss / spe).item())
        if log:
            print(f"  epoch {ep+1:4d}/{epochs}  loss={hist[-1]:.4f}  lr={opt.lr:.1e}", flush=True)
    return model, np.array(hist)


def extract_features(model, imgs, batch=256):
    feats = []
    for i in range(0, len(imgs), batch):
        feats.append(model.features(M._t(imgs[i:i + batch])).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def linear_probe(Xtr, ytr, Xte, yte, n_classes=10, epochs=200, batch=256, lr=1e-2, seed=0,
                 return_clf=False):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtrn, Xten = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(seed)
    clf = M.Linear(Xtrn.shape[1], n_classes)
    opt = M.Adam(clf.params(), lr=lr, wd=1e-4)
    Xt = M._t(Xtrn); yt = torch.as_tensor(np.asarray(ytr), device=M.DEV, dtype=torch.long)
    n = len(Xtrn)
    for ep in range(epochs):
        order = torch.randperm(n, device=M.DEV)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb, yb = Xt[idx], yt[idx]
            p = M.softmax(clf.forward(xb), -1)
            oneh = torch.zeros_like(p); oneh[torch.arange(len(yb), device=M.DEV), yb] = 1
            clf.backward((p - oneh) / len(yb)); opt.step()
    pred = clf.forward(M._t(Xten)).argmax(1).cpu().numpy()
    acc = float((pred == np.asarray(yte)).mean())
    if return_clf:
        return acc, clf, mu, sd
    return acc


def make_reconstruction(model, imgs, mask_ratio=0.75, seed=0):
    torch.manual_seed(seed)
    x = M._t(imgs)
    latent, mask, ids_restore = model.encoder.forward(x, mask_ratio)
    pred = model.decoder.forward(latent, ids_restore)
    P = model.cfg["P"]
    target = M.patchify(x, P)
    if model.cfg.get("norm_pix", True):
        mu, std = M.patch_stats(x, P)
        pred = pred * std + mu
    m = mask[:, :, None]
    masked = target * (1 - m) + 0.5 * m
    recon = target * (1 - m) + pred * m
    to = lambda t: M.unpatchify(t, P).clamp(0, 1).detach().cpu().numpy()
    return to(target), to(masked), to(recon)


def overfit_one_batch(cfg, x, steps=200, lr=1e-3):
    model = M.MAE(cfg)
    opt = M.Adam(model.params(), lr=lr)
    X = M._t(x); hist = []
    for _ in range(steps):
        torch.manual_seed(0)
        loss, _, _, _ = model.forward(X)
        model.backward(); opt.step()
        hist.append(loss.item())
    return np.array(hist)
