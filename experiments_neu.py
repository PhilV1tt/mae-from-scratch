# MAE auto-supervise sur NEU (defauts de surface d'acier), puis classification des 6 defauts
# par sonde lineaire. Etude de l'effet du nombre d'etiquettes. Backend torch sur GPU (MPS).

import json
import numpy as np
import torch
import mae as M
import train as T

M.use("mps", torch.float32)

CFG = dict(img=96, P=8, C=3, D=128, enc_depth=6, enc_heads=4,
           dec_dim=64, dec_depth=2, dec_heads=4, mlp_ratio=4, norm_pix=True)
EPOCHS = 800
BATCH = 256
BASE_LR = 8e-4
MASK = 0.75
PER_CLASS = [10, 40, 120, 240]
RES = "results"


def probe_subsets(Ftr, ytr, Fte, yte, per_class):
    rng = np.random.RandomState(0)
    out = {}
    for k in per_class:
        idx = []
        for c in range(6):
            ci = np.where(ytr == c)[0]; rng.shuffle(ci); idx += list(ci[:k])
        idx = np.array(idx)
        out[k] = T.linear_probe(Ftr[idx], ytr[idx], Fte, yte, n_classes=6, epochs=300)
    return out


def main():
    np.random.seed(0); torch.manual_seed(0)
    print("chargement NEU...", flush=True)
    Xtr, ytr, Xte, yte, classes = T.load_neu()
    print(f"  {len(Xtr)} train / {len(Xte)} test, 6 classes", flush=True)

    base = M.MAE(CFG)
    Ftr0, Fte0 = T.extract_features(base, Xtr), T.extract_features(base, Xte)
    acc_base = T.linear_probe(Ftr0, ytr, Fte0, yte, n_classes=6, epochs=400)
    print(f"reference encodeur aleatoire : acc={acc_base:.3f}", flush=True)

    print(f"pretraining MAE mask={MASK} ...", flush=True)
    model, hist = T.pretrain(CFG, Xtr, epochs=EPOCHS, batch=BATCH,
                             base_lr=BASE_LR, mask_ratio=MASK, seed=0, log=False)
    print(f"  loss {hist[0]:.3f} -> {hist[-1]:.3f}", flush=True)
    Ftr, Fte = T.extract_features(model, Xtr), T.extract_features(model, Xte)

    acc_full, clf, mu, sd = T.linear_probe(Ftr, ytr, Fte, yte, n_classes=6, epochs=400, return_clf=True)
    print(f"MAE sonde lineaire (toutes etiquettes) : acc={acc_full:.3f}  (ref {acc_base:.3f})", flush=True)

    mae_eff = probe_subsets(Ftr, ytr, Fte, yte, PER_CLASS)
    rnd_eff = probe_subsets(Ftr0, ytr, Fte0, yte, PER_CLASS)
    print("efficacite (k/classe : MAE | aleatoire)", flush=True)
    for k in PER_CLASS:
        print(f"  {k:3d} : {mae_eff[k]:.3f} | {rnd_eff[k]:.3f}", flush=True)

    M.save_params(model, f"{RES}/neu_model.npz")
    np.savez(f"{RES}/neu_probe.npz", W=clf.W.detach().cpu().numpy(), b=clf.b.detach().cpu().numpy(),
             mu=mu, sd=sd, names=np.array(classes))
    orig, masked, recon = T.make_reconstruction(model, Xte[:8], mask_ratio=MASK, seed=0)
    np.savez(f"{RES}/neu_recon.npz", orig=orig, masked=masked, recon=recon)
    np.save(f"{RES}/neu_loss.npy", hist)
    json.dump({"acc_full": acc_full, "acc_base": acc_base, "per_class": PER_CLASS,
               "mae_eff": mae_eff, "rnd_eff": rnd_eff,
               "loss_start": float(hist[0]), "loss_end": float(hist[-1]),
               "n_train": len(Xtr), "n_test": len(Xte), "classes": classes},
              open(f"{RES}/neu.json", "w"), indent=2)
    print("NEU termine.", flush=True)


if __name__ == "__main__":
    main()
