# Genere les figures du rapport depuis results/. A lancer apres les experiences.

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mae as M, train as T

FIG = "report/figures"
BLUE = "#3b6fb0"


def fig_recon(npz, path, mask_label="masqué 75%"):
    rec = np.load(npz)
    orig, masked, recon = rec["orig"], rec["masked"], rec["recon"]
    n = min(6, len(orig))
    fig, ax = plt.subplots(3, n, figsize=(1.8 * n, 5.4))
    for j in range(n):
        for i, (im, lab) in enumerate([(orig, "original"), (masked, mask_label), (recon, "reconstruit")]):
            ax[i, j].imshow(im[j], cmap="gray"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0:
                ax[i, 0].set_ylabel(lab, fontsize=11)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# STL-10
if os.path.exists("results/ablation.json"):
    res = json.load(open("results/ablation.json"))
    lc = np.load("results/loss_curves.npz")
    ratios = sorted(res["results"], key=float)

    plt.figure(figsize=(6, 4))
    for r in res["ratios"]:
        plt.plot(lc[str(r)], label=f"mask {r}")
    plt.xlabel("époque"); plt.ylabel("perte MSE (patchs masqués)")
    plt.title("STL-10, perte de reconstruction")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/loss_curves.png", dpi=130); plt.close()

    if os.path.exists("results/recon.npz"):
        fig_recon("results/recon.npz", f"{FIG}/reconstruction.png")

    accs = [res["results"][r]["acc"] for r in ratios]
    base = res["baseline_acc"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([f"mask {r}" for r in ratios] + ["aléatoire", "hasard"], accs + [base, 0.1],
              color=[BLUE] * len(ratios) + ["gray", "lightgray"])
    ax[0].set_ylabel("exactitude"); ax[0].set_title("Sonde linéaire vs références")
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].plot([float(r) for r in ratios], accs, "o-", color=BLUE)
    ax[1].axhline(base, ls="--", color="gray", label="encodeur aléatoire")
    ax[1].set_xlabel("taux de masquage"); ax[1].set_ylabel("exactitude sonde linéaire")
    ax[1].set_title("Ablation du taux de masquage"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/probe_ablation.png", dpi=130); plt.close()
    print("STL : baseline %.3f | " % base
          + "  ".join(f"{r}:{res['results'][r]['acc']:.3f}" for r in ratios))

# NEU
if os.path.exists("results/neu.json"):
    neu = json.load(open("results/neu.json"))
    if os.path.exists("results/neu_recon.npz"):
        fig_recon("results/neu_recon.npz", f"{FIG}/neu_reconstruction.png")

    pc = neu["per_class"]; totals = [k * 6 for k in pc]
    plt.figure(figsize=(6, 4))
    plt.plot(totals, [neu["mae_eff"][str(k)] for k in pc], "o-", color=BLUE, label="MAE pré-entraîné")
    plt.plot(totals, [neu["rnd_eff"][str(k)] for k in pc], "s--", color="gray", label="encodeur aléatoire")
    plt.xlabel("nombre d'images étiquetées"); plt.ylabel("exactitude (6 défauts)")
    plt.title("NEU, efficacité en annotations")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/neu_efficiency.png", dpi=130); plt.close()

    # matrice de confusion
    if os.path.exists("results/neu_model.npz") and os.path.exists("results/neu_probe.npz"):
        import experiments_neu as E
        model = M.load_params(M.MAE(E.CFG), "results/neu_model.npz")
        probe = np.load("results/neu_probe.npz", allow_pickle=True)
        names = [str(s) for s in probe["names"]]
        _, _, Xte, yte, _ = T.load_neu()
        F = T.extract_features(model, Xte)
        F = (F - probe["mu"]) / probe["sd"]
        pred = (F @ probe["W"] + probe["b"]).argmax(1)
        K = len(names); cm = np.zeros((K, K), int)
        for t, p in zip(yte, pred):
            cm[t, p] += 1
        plt.figure(figsize=(5, 4.4)); plt.imshow(cm, cmap="Blues")
        for i in range(K):
            for j in range(K):
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        short = [n[:5] for n in names]
        plt.xticks(range(K), short, rotation=45, fontsize=8); plt.yticks(range(K), short, fontsize=8)
        plt.xlabel("prédit"); plt.ylabel("vrai"); plt.title("NEU, matrice de confusion")
        plt.tight_layout(); plt.savefig(f"{FIG}/neu_confusion.png", dpi=130); plt.close()
    print("NEU : ref %.3f | MAE %.3f" % (neu["acc_base"], neu["acc_full"]))

print("ok.")
