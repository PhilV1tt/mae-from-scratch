# Ablation du mask ratio sur STL-10 : pretraining MAE + linear probe pour chaque ratio.
# Backend torch sur GPU (MPS). Sauvegarde courbes de loss, modele 0.75, reconstructions, probe.

import json
import numpy as np
import torch
import mae as M
import train as T

M.use("mps", torch.float32)

CFG = dict(img=96, P=8, C=3, D=128, enc_depth=6, enc_heads=4,
           dec_dim=64, dec_depth=2, dec_heads=4, mlp_ratio=4, norm_pix=True)
N_PRE = 12000
EPOCHS = 100
BATCH = 256
BASE_LR = 8e-4
RATIOS = [0.5, 0.25, 0.75, 0.9]
MAIN = 0.75
RES = "results"


def class_names():
    with open(f"{T.DATA}/class_names.txt") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    np.random.seed(0); torch.manual_seed(0)
    print("chargement des donnees...", flush=True)
    pre, _ = T.load_stl("unlabeled", N_PRE)
    Xtr_i, ytr = T.load_stl("train", 5000)
    Xte_i, yte = T.load_stl("test", 8000)

    base = M.MAE(CFG)
    Ftr0, Fte0 = T.extract_features(base, Xtr_i), T.extract_features(base, Xte_i)
    base_acc = T.linear_probe(Ftr0, ytr, Fte0, yte)
    print(f"baseline encodeur aleatoire : acc={base_acc:.3f}", flush=True)

    out = {"config": CFG, "n_pre": N_PRE, "epochs": EPOCHS, "base_lr": BASE_LR,
           "baseline_acc": base_acc, "ratios": [], "results": {}}
    losses = {}

    for r in RATIOS:
        print(f"\n=== mask_ratio={r} ===", flush=True)
        model, hist = T.pretrain(CFG, pre, epochs=EPOCHS, batch=BATCH,
                                 base_lr=BASE_LR, mask_ratio=r, seed=0, log=False)
        print(f"  loss {hist[0]:.3f} -> {hist[-1]:.3f}", flush=True)
        Ftr, Fte = T.extract_features(model, Xtr_i), T.extract_features(model, Xte_i)
        acc, clf, pmu, psd = T.linear_probe(Ftr, ytr, Fte, yte, return_clf=True)
        print(f"  linear probe acc={acc:.3f}  (baseline {base_acc:.3f}, gain {acc-base_acc:+.3f})", flush=True)

        out["ratios"].append(r)
        out["results"][str(r)] = {"acc": acc, "final_loss": float(hist[-1])}
        losses[str(r)] = hist

        if r == MAIN:
            M.save_params(model, f"{RES}/model_main.npz")
            np.savez(f"{RES}/probe_main.npz", W=clf.W.detach().cpu().numpy(),
                     b=clf.b.detach().cpu().numpy(), mu=pmu, sd=psd, names=np.array(class_names()))
            orig, masked, recon = T.make_reconstruction(model, Xte_i[:8], mask_ratio=MAIN, seed=0)
            np.savez(f"{RES}/recon.npz", orig=orig, masked=masked, recon=recon)

        with open(f"{RES}/ablation.json", "w") as f:
            json.dump(out, f, indent=2)
        np.savez(f"{RES}/loss_curves.npz", **losses)
        torch.mps.empty_cache()

    print("\nrecapitulatif")
    print(f"  baseline {base_acc:.3f}")
    for r in RATIOS:
        rr = out["results"][str(r)]
        print(f"  ratio {r}: acc={rr['acc']:.3f}  loss_fin={rr['final_loss']:.3f}")
    print("termine.", flush=True)


if __name__ == "__main__":
    main()
