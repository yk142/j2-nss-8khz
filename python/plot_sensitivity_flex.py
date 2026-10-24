"""感度実験の結果図: (a)(b) の誤差と (c)(c3) の関係（学習済み A・B の点を重ねる）。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("reports/issue-34/sensitivity.json"))
# 学習済みモデル（#33 のスクリーニング、(b) は θ RMSE、(c3) は reports/issue-34/c3_summary.json）
trained = {
    "A w64": (0.128, 0.0062, 21, 0.985), "A w128": (0.100, 0.0068, 21, 0.979),
    "A w256": (0.086, 0.0117, 21, 0.988), "B w64": (0.024, 0.0047, 21, 0.997),
    "B w128": (0.022, 0.0050, 21, 0.997), "B w256": (0.020, 0.0049, 21, 0.997),
}
fam = {"k": "tab:red", "c": "tab:orange", "Jm": "tab:blue", "JL": "tab:cyan",
       "mgL": "tab:green", "Fc": "tab:purple", "Bv": "tab:brown", "nominal": "k"}
def family(n):
    for f in ("nominal", "mgL", "Jm", "JL", "Fc", "Bv", "k", "c"):
        if n.startswith(f):
            return f

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for r in d:
    f = family(r["name"]); col = fam[f]
    for a, y in zip(ax, (r["c_worst_deg"], 21 - r["c3_match"], r["c3_spearman_J"])):
        a.scatter(r["b_q_deg"], y, c=col, s=40)
        a.annotate(r["name"], (r["b_q_deg"], y), fontsize=7, xytext=(3, 3), textcoords="offset points")
for n, (b, c, m, rho) in trained.items():
    for a, y in zip(ax, (c, 21 - m, rho)):
        a.scatter(b, y, marker="*", c="gold", edgecolors="k", s=160, zorder=5)
    ax[0].annotate(n, (b, c), fontsize=7, xytext=(3, -10), textcoords="offset points")
for f, col in fam.items():
    ax[0].scatter([], [], c=col, label=f)
ax[0].scatter([], [], marker="*", c="gold", edgecolors="k", s=100, label="trained A/B")
ax[0].legend(fontsize=7)
ax[0].axhline(0.1, c="gray", ls="--"); ax[0].set_yscale("log"); ax[0].set_ylabel("(c) worst theta RMSE [deg]")
ax[1].set_ylabel("(c3) misclassified gains (of 21)")
ax[2].axhline(0.9, c="gray", ls="--"); ax[2].set_ylabel("(c3) Spearman of J")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("(b) 4000-step window theta RMSE [deg]"); a.grid(alpha=.3)
ax[0].axvline(0.4, c="gray", ls=":")
fig.tight_layout(); fig.savefig("reports/issue-34/plots/sensitivity_ab_to_c.png", dpi=130)
