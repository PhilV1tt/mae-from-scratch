# MAE implemente entierement a la main pour comprendre le modele en detail.
# torch ne sert que de bibliotheque de tableaux (pour utiliser le GPU) : aucune fonction
# toute faite, ni autograd ni torch.nn. Tout le forward et le backward sont ecrits a la main.
# DEV/DT fixent l'appareil et la precision : cpu/float64 pour le gradient check,
# mps/float32 pour l'entrainement. var(...) utilise unbiased=False (variance population).

import numpy as np
import torch

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DT = torch.float32


def use(device, dtype):
    global DEV, DT
    DEV, DT = torch.device(device), dtype


def _t(x):
    return torch.as_tensor(x, device=DEV, dtype=DT)


# geometrie
def patchify(imgs, P=16):
    B, H, W, C = imgs.shape
    g = H // P
    x = imgs.reshape(B, g, P, g, P, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, g * g, P * P * C)


def unpatchify(patches, P=16):
    B, N, pd = patches.shape
    C = pd // (P * P)
    g = int(round(N ** 0.5))
    x = patches.reshape(B, g, g, P, P, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, g * P, g * P, C)


def pos_embed_1d(positions, d):
    k = torch.arange(d // 2, device=DEV, dtype=DT)
    denom = 10000.0 ** (2 * k / d)
    args = positions[:, None].to(DT) / denom[None, :]
    pe = torch.zeros((len(positions), d), device=DEV, dtype=DT)
    pe[:, 0::2] = torch.sin(args)
    pe[:, 1::2] = torch.cos(args)
    return pe


def pos_embed_2d(N, D):
    g = int(round(N ** 0.5))
    i = torch.arange(N, device=DEV)
    return torch.cat([pos_embed_1d(i // g, D // 2), pos_embed_1d(i % g, D // 2)], dim=1)


# primitives (softmax/gelu ecrits a la main)
def softmax(x, axis=-1):
    x = x - x.amax(dim=axis, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=axis, keepdim=True)


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh((2.0 / np.pi) ** 0.5 * (x + 0.044715 * x ** 3)))


def gelu_backward(x):
    K = (2.0 / np.pi) ** 0.5
    u = K * (x + 0.044715 * x ** 3)
    t = torch.tanh(u)
    du = K * (1.0 + 0.134145 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def random_masking(x, mask_ratio):
    B, N, D = x.shape
    N_vis = int(round(N * (1 - mask_ratio)))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :N_vis]
    x_vis = torch.gather(x, 1, ids_keep[:, :, None].expand(B, N_vis, D))
    mask = torch.ones((B, N), device=x.device, dtype=x.dtype)
    mask[:, :N_vis] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_vis, mask, ids_restore


# couches
class Linear:
    def __init__(self, n_in, n_out):
        self.W = torch.randn(n_in, n_out, device=DEV, dtype=DT) * (2.0 / n_in) ** 0.5
        self.b = torch.zeros(n_out, device=DEV, dtype=DT)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dy):
        x = self.cache
        self.dW = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        self.db = dy.reshape(-1, dy.shape[-1]).sum(0)
        return dy @ self.W.T

    def params(self):
        return [(self, "W"), (self, "b")]


class LayerNorm:
    def __init__(self, D, eps=1e-5):
        self.gamma = torch.ones(D, device=DEV, dtype=DT)
        self.beta = torch.zeros(D, device=DEV, dtype=DT)
        self.eps = eps
        self.cache = None

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        xhat = (x - mu) / std
        self.cache = (xhat, std)
        return self.gamma * xhat + self.beta

    def backward(self, dy):
        xhat, std = self.cache
        D = xhat.shape[-1]
        self.dgamma = (dy * xhat).reshape(-1, D).sum(0)
        self.dbeta = dy.reshape(-1, D).sum(0)
        dxhat = dy * self.gamma
        dx = (dxhat
              - dxhat.mean(-1, keepdim=True)
              - xhat * (dxhat * xhat).mean(-1, keepdim=True)) / std
        return dx

    def params(self):
        return [(self, "gamma"), (self, "beta")]


class GELU:
    def forward(self, x):
        self.cache = x
        return gelu(x)

    def backward(self, dy):
        return dy * gelu_backward(self.cache)

    def params(self):
        return []


class MHA:
    def __init__(self, D, n_heads):
        assert D % n_heads == 0
        self.h = n_heads
        self.dk = D // n_heads
        self.q = Linear(D, D)
        self.k = Linear(D, D)
        self.v = Linear(D, D)
        self.o = Linear(D, D)

    def _split(self, x):
        B, N, _ = x.shape
        return x.reshape(B, N, self.h, self.dk).permute(0, 2, 1, 3)

    def _merge(self, x):
        B, h, N, dk = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, N, h * dk)

    def forward(self, x):
        Q = self._split(self.q.forward(x))
        K = self._split(self.k.forward(x))
        V = self._split(self.v.forward(x))
        scale = 1.0 / self.dk ** 0.5
        S = (Q @ K.transpose(-1, -2)) * scale
        P = softmax(S, -1)
        O = P @ V
        out = self.o.forward(self._merge(O))
        self.cache = (Q, K, V, P, scale)
        return out

    def backward(self, dout):
        Q, K, V, P, scale = self.cache
        dO = self._split(self.o.backward(dout))
        dP = dO @ V.transpose(-1, -2)
        dV = P.transpose(-1, -2) @ dO
        dS = P * (dP - (dP * P).sum(-1, keepdim=True)) * scale
        dQ = dS @ K
        dK = dS.transpose(-1, -2) @ Q
        dx = (self.q.backward(self._merge(dQ))
              + self.k.backward(self._merge(dK))
              + self.v.backward(self._merge(dV)))
        return dx

    def params(self):
        return self.q.params() + self.k.params() + self.v.params() + self.o.params()


class MLP:
    def __init__(self, D, ratio=4):
        self.fc1 = Linear(D, ratio * D)
        self.act = GELU()
        self.fc2 = Linear(ratio * D, D)

    def forward(self, x):
        return self.fc2.forward(self.act.forward(self.fc1.forward(x)))

    def backward(self, dy):
        return self.fc1.backward(self.act.backward(self.fc2.backward(dy)))

    def params(self):
        return self.fc1.params() + self.fc2.params()


class Block:
    def __init__(self, D, n_heads, ratio=4):
        self.ln1 = LayerNorm(D)
        self.mha = MHA(D, n_heads)
        self.ln2 = LayerNorm(D)
        self.mlp = MLP(D, ratio)

    def forward(self, x):
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, dx):
        dx = dx + self.ln2.backward(self.mlp.backward(dx))
        dx = dx + self.ln1.backward(self.mha.backward(dx))
        return dx

    def params(self):
        return self.ln1.params() + self.mha.params() + self.ln2.params() + self.mlp.params()


# encodeur / decodeur
class Encoder:
    def __init__(self, cfg):
        D = cfg["D"]
        pd = cfg["P"] ** 2 * cfg["C"]
        self.N = (cfg["img"] // cfg["P"]) ** 2
        self.patch_embed = Linear(pd, D)
        self.pos = pos_embed_2d(self.N, D)
        self.blocks = [Block(D, cfg["enc_heads"], cfg["mlp_ratio"]) for _ in range(cfg["enc_depth"])]
        self.ln = LayerNorm(D)
        self.P = cfg["P"]

    def forward(self, imgs, mask_ratio):
        x = self.patch_embed.forward(patchify(imgs, self.P)) + self.pos[None]
        B, N, D = x.shape
        x_vis, mask, ids_restore = random_masking(x, mask_ratio)
        N_vis = x_vis.shape[1]
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :N_vis]
        h = x_vis
        for blk in self.blocks:
            h = blk.forward(h)
        latent = self.ln.forward(h)
        self.cache = (ids_keep, N, D)
        return latent, mask, ids_restore

    def backward(self, dlatent):
        ids_keep, N, D = self.cache
        dh = self.ln.backward(dlatent)
        for blk in reversed(self.blocks):
            dh = blk.backward(dh)
        B, N_vis, _ = dh.shape
        dx = torch.zeros((B, N, D), device=dh.device, dtype=dh.dtype)
        dx.scatter_(1, ids_keep[:, :, None].expand(B, N_vis, D), dh)
        self.patch_embed.backward(dx)
        return None

    def params(self):
        out = self.patch_embed.params()
        for blk in self.blocks:
            out += blk.params()
        return out + self.ln.params()


class Decoder:
    def __init__(self, cfg):
        D, dec = cfg["D"], cfg["dec_dim"]
        pd = cfg["P"] ** 2 * cfg["C"]
        self.N = (cfg["img"] // cfg["P"]) ** 2
        self.embed = Linear(D, dec)
        self.mask_token = torch.randn(dec, device=DEV, dtype=DT) * 0.02
        self.pos = pos_embed_2d(self.N, dec)
        self.blocks = [Block(dec, cfg["dec_heads"], cfg["mlp_ratio"]) for _ in range(cfg["dec_depth"])]
        self.ln = LayerNorm(dec)
        self.pred = Linear(dec, pd)
        self.dec = dec

    def forward(self, latent, ids_restore):
        B, N_vis, _ = latent.shape
        N = ids_restore.shape[1]
        v = self.embed.forward(latent)
        mt = self.mask_token.expand(B, N - N_vis, self.dec)
        seq_ = torch.cat([v, mt], dim=1)
        seq = torch.gather(seq_, 1, ids_restore[:, :, None].expand(B, N, self.dec))
        h = seq + self.pos[None]
        for blk in self.blocks:
            h = blk.forward(h)
        pred = self.pred.forward(self.ln.forward(h))
        self.cache = (N_vis, N, ids_restore)
        return pred

    def backward(self, dpred):
        N_vis, N, ids_restore = self.cache
        d = self.ln.backward(self.pred.backward(dpred))
        for blk in reversed(self.blocks):
            d = blk.backward(d)
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        dseq_ = torch.gather(d, 1, ids_shuffle[:, :, None].expand(d.shape[0], N, self.dec))
        dv = dseq_[:, :N_vis, :]
        self.dmask_token = dseq_[:, N_vis:, :].sum(dim=(0, 1))
        return self.embed.backward(dv)

    def params(self):
        out = self.embed.params() + [(self, "mask_token")]
        for blk in self.blocks:
            out += blk.params()
        return out + self.ln.params() + self.pred.params()


# perte
def mae_loss_and_grad(imgs, pred, mask, norm_pix=True, P=16, eps=1e-6):
    target = patchify(imgs, P)
    if norm_pix:
        mu = target.mean(-1, keepdim=True)
        var = target.var(-1, keepdim=True, unbiased=False)
        target = (target - mu) / torch.sqrt(var + eps)
    diff = pred - target
    Dp = pred.shape[-1]
    n_masked = mask.sum()
    loss = ((diff ** 2).mean(-1) * mask).sum() / n_masked
    dpred = (2.0 / Dp) * diff * mask[:, :, None] / n_masked
    return loss, dpred


def patch_stats(imgs, P=16, eps=1e-6):
    target = patchify(imgs, P)
    mu = target.mean(-1, keepdim=True)
    std = torch.sqrt(target.var(-1, keepdim=True, unbiased=False) + eps)
    return mu, std


# modele complet
class MAE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.cfg["mask_ratio"]
        latent, mask, ids_restore = self.encoder.forward(imgs, mask_ratio)
        pred = self.decoder.forward(latent, ids_restore)
        loss, dpred = mae_loss_and_grad(imgs, pred, mask,
                                        norm_pix=self.cfg.get("norm_pix", True), P=self.cfg["P"])
        self._dpred = dpred
        return loss, pred, mask, ids_restore

    def backward(self):
        self.encoder.backward(self.decoder.backward(self._dpred))

    def params(self):
        return self.encoder.params() + self.decoder.params()

    def features(self, imgs):
        latent, _, _ = self.encoder.forward(imgs, 0.0)
        return latent.mean(dim=1)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [torch.zeros_like(getattr(o, n)) for o, n in params]
        self.v = [torch.zeros_like(getattr(o, n)) for o, n in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (o, n) in enumerate(self.params):
            g = getattr(o, "d" + n)
            p = getattr(o, n)
            if self.wd and p.ndim >= 2:
                p.mul_(1.0 - self.lr * self.wd)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.add_(-self.lr * mhat / (torch.sqrt(vhat) + self.eps))


def default_config():
    return dict(img=96, P=16, C=3, D=128, enc_depth=4, enc_heads=4,
                dec_dim=64, dec_depth=2, dec_heads=4, mlp_ratio=4,
                mask_ratio=0.75, norm_pix=True)


def save_params(model, path):
    np.savez(path, *[getattr(o, n).detach().cpu().numpy() for o, n in model.params()])


def load_params(model, path):
    d = np.load(path)
    for (o, n), k in zip(model.params(), d.files):
        setattr(o, n, _t(d[k]))
    return model
