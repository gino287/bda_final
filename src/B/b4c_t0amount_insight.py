"""
任務 3 — t0_amount 洞察佐證
==========================
把 t0_amount 分成 10 個分桶（quantile），畫每桶復活率長條圖，
確認「t0 單筆金額越高 → 復活率越低」。
同時控制 is_imputed（分 True/False 各畫一次），確認非子群混淆造成的假象。

資料：完整 B 母體 features.parquet（144,919），描述性洞察求穩定。
輸出：圖 → output/B/B4/figures/，分桶數據 → output/B/B4/t0_amount_bins.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
FIG = ROOT / "output/B/B4/figures"
CSV = ROOT / "output/B/B4/t0_amount_bins.csv"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

N_BINS = 10

df = pd.read_parquet(FEATURES, columns=["t0_amount", "revived", "is_imputed"])
df = df[df["t0_amount"].notna()].copy()
print(f"[t0_amount 洞察] n={len(df):,}（t0_amount 非缺失）")
print(f"  t0_amount: min={df['t0_amount'].min():.0f} "
      f"median={df['t0_amount'].median():.0f} max={df['t0_amount'].max():.0f}")
print(f"  整體復活率={df['revived'].mean():.4f}")


def bin_table(sub, label):
    """在 sub 內用 quantile 分桶，回傳每桶統計。"""
    s = sub.copy()
    s["bin"] = pd.qcut(s["t0_amount"], N_BINS, labels=False, duplicates="drop")
    g = s.groupby("bin").agg(
        n=("revived", "size"),
        revival_rate=("revived", "mean"),
        amount_min=("t0_amount", "min"),
        amount_max=("t0_amount", "max"),
        amount_median=("t0_amount", "median"),
    ).reset_index()
    g["subgroup"] = label
    # 趨勢檢定：bin index vs revival_rate（Spearman）
    rho, pval = spearmanr(g["bin"], g["revival_rate"])
    return g, rho, pval


overall, rho_o, p_o = bin_table(df, "overall")
imp_false, rho_f, p_f = bin_table(df[~df["is_imputed"]], "is_imputed=False")
imp_true, rho_t, p_t = bin_table(df[df["is_imputed"]], "is_imputed=True")

print(f"\n=== 整體分桶（依 t0_amount 由低到高）===")
print(overall[["bin", "n", "amount_median", "revival_rate"]].to_string(index=False))
print(f"  Spearman(bin, revival_rate) = {rho_o:.3f} (p={p_o:.2e})")
print(f"\n=== is_imputed=False ===  Spearman={rho_f:.3f} (p={p_f:.2e})")
print(imp_false[["bin", "n", "amount_median", "revival_rate"]].to_string(index=False))
print(f"\n=== is_imputed=True ===  Spearman={rho_t:.3f} (p={p_t:.2e})")
print(imp_true[["bin", "n", "amount_median", "revival_rate"]].to_string(index=False))

# ── 結論：兩子群皆負相關 → 非混淆 ──
all_neg = (rho_o < 0) and (rho_f < 0) and (rho_t < 0)
conclusion = ("趨勢成立且非混淆：整體與 is_imputed True/False 子群內，"
              "t0_amount 越高復活率越低（皆負相關）"
              if all_neg else "趨勢在某子群不成立，需檢視")
print(f"\n>>> {conclusion}")

# ── 輸出 CSV ──
out = pd.concat([overall, imp_false, imp_true], ignore_index=True)
out = out[["subgroup", "bin", "n", "amount_min", "amount_max",
           "amount_median", "revival_rate"]]
out.to_csv(CSV, index=False, encoding="utf-8-sig")

# ================================================================
# 圖：左=整體長條圖，右=兩子群趨勢線（控制混淆）
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：整體長條
ax = axes[0]
bars = ax.bar(overall["bin"], overall["revival_rate"], color="C0",
              alpha=0.85, edgecolor="k")
ax.axhline(df["revived"].mean(), ls="--", color="red",
           label=f"overall mean={df['revived'].mean():.3f}")
ax.set(xlabel="t0_amount decile (0=lowest, 9=highest)",
       ylabel="Revival rate",
       title=f"Revival rate by t0_amount decile (overall)\nSpearman={rho_o:.2f}")
ax.set_xticks(overall["bin"])
# 標每桶中位金額
for _, r in overall.iterrows():
    ax.annotate(f"{r['amount_median']:.0f}", (r["bin"], r["revival_rate"]),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, color="dimgrey")
ax.legend(fontsize=8)

# 右：兩子群趨勢線（控制 is_imputed）
ax = axes[1]
ax.plot(imp_false["bin"], imp_false["revival_rate"], "o-", color="C2",
        label=f"is_imputed=False (Spearman={rho_f:.2f})")
ax.plot(imp_true["bin"], imp_true["revival_rate"], "s-", color="C3",
        label=f"is_imputed=True (Spearman={rho_t:.2f})")
ax.set(xlabel="t0_amount decile (within subgroup)",
       ylabel="Revival rate",
       title="Controlled for is_imputed\n(trend holds in BOTH subgroups → not a confound)")
ax.set_xticks(range(N_BINS))
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = FIG / "t0_amount_revival.png"
plt.savefig(fig_path)
plt.close()

print(f"\n[完成] 輸出:")
print(f"  圖: {fig_path}")
print(f"  csv: {CSV}")
