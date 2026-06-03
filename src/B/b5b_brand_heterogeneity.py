"""
任務 2 — 品牌異質性表
=====================
把時間外 test 集按 ShopId 拆開，每品牌算：
  n、base rate、ROC-AUC、PR-AUC、第1分位 lift、profit-max 觸及比例、模型 vs 全發券利潤差。
依 base rate 排序，輸出 csv + 圖（x=base rate, y=lift / 利潤增益），
驗證「低復活率品牌 targeting 價值最大」。

決策機率：校準後 LightGBM（p_lightgbm_cal）。
Profit 用任務 1 的 Uplift 模型（u 代表值，取 config cannibalization_uplift=0.2）：
  原版經濟（m/c≈11）下 profit-max≈全發券，看不出品牌差異；
  uplift 模型才能呈現 targeting 的真實價值。
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
CONFIG = ROOT / "config/unit_economics.yaml"
RA = ROOT / "output/B/B5/report_assets"
RA.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# ── 假設 ──
cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
gm = cfg["gross_margin"]
coupon_cost = cfg["coupon_face_value"] * cfg["redemption_rate"]
contact_cost = coupon_cost + cfg["fixed_contact_cost"]   # 每位被觸及成本（uplift 模型）
u = cfg["revised_profit"]["cannibalization_uplift"]      # 代表性 uplift = 0.2

preds = pd.read_parquet(PREDS)
print(f"[品牌異質性] test n={len(preds):,}, 代表 uplift u={u}, "
      f"gm={gm}, contact_cost={contact_cost:.0f}")

taus = np.linspace(0, 1, 201)


def decile1_lift(y, p):
    order = np.argsort(-p)
    ys = y[order]
    k = max(1, len(ys) // 10)
    return float(ys[:k].mean() / y.mean()) if y.mean() > 0 else np.nan


def uplift_profit(treated, y, rev):
    """Uplift 模型：只有 u 比例復活營收算增量，成本加在所有被觸及者。"""
    n_t = treated.sum()
    inc = u * gm * rev[treated & (y == 1)].sum()
    return inc - contact_cost * n_t


rows = []
for shop, g in preds.groupby("ShopId"):
    y = g["revived"].values.astype(float)
    p = g["p_lightgbm_cal"].values
    rev = g["future_90d_revenue"].values.astype(float)
    n = len(g)
    base = y.mean()
    # 需兩類齊全才算 AUC
    if y.min() == y.max():
        auc = prauc = np.nan
    else:
        auc = roc_auc_score(y, p)
        prauc = average_precision_score(y, p)
    lift1 = decile1_lift(y, p)

    # profit-max（模型 τ 掃描） vs 全發券
    profits = np.array([uplift_profit(p >= t, y, rev) for t in taus])
    bi = int(np.argmax(profits))
    tau_star = float(taus[bi])
    model_profit = float(profits[bi])
    n_treat = int((p >= tau_star).sum())
    all_profit = uplift_profit(np.ones(n, bool), y, rev)
    gap = model_profit - all_profit

    rows.append({
        "ShopId": shop,
        "ShopId_short": shop[:8],
        "n": n,
        "base_rate": round(base, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "pr_auc": round(prauc, 4) if not np.isnan(prauc) else np.nan,
        "decile1_lift": round(lift1, 3),
        "profitmax_reach_ratio": round(n_treat / n, 3),
        "model_profit": round(model_profit, 0),
        "treat_all_profit": round(all_profit, 0),
        "model_vs_all_gap": round(gap, 0),
        "gap_per_member": round(gap / n, 1),
    })

brand = pd.DataFrame(rows).sort_values("base_rate").reset_index(drop=True)

print("\n=== 品牌異質性表（依 base rate 排序）===")
print(brand.drop(columns=["ShopId"]).to_string(index=False))

# ── 相關性驗證：低 base rate ↔ 高 targeting 價值 ──
corr_lift = brand["base_rate"].corr(brand["decile1_lift"])
corr_gap = brand["base_rate"].corr(brand["gap_per_member"])
print(f"\n相關係數: base_rate vs decile1_lift = {corr_lift:.3f}")
print(f"相關係數: base_rate vs gap_per_member = {corr_gap:.3f}")
conclusion = ("驗證成立：base rate 越低，lift 與每人利潤增益越高（負相關）"
              if (corr_lift < 0 and corr_gap < 0)
              else "部分成立，見表")
print(f">>> {conclusion}")

# ── 輸出 CSV ──
csv_path = RA / "brand_heterogeneity.csv"
brand.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ── 圖：x=base_rate，雙面板（lift / 每人利潤增益）──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sizes = (brand["n"] / brand["n"].max() * 600 + 60)

# 左：base_rate vs decile1 lift
ax = axes[0]
ax.scatter(brand["base_rate"], brand["decile1_lift"], s=sizes,
           c=brand["base_rate"], cmap="coolwarm_r", edgecolor="k", zorder=3)
for _, r in brand.iterrows():
    ax.annotate(r["ShopId_short"], (r["base_rate"], r["decile1_lift"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
# 趨勢線
z = np.polyfit(brand["base_rate"], brand["decile1_lift"], 1)
xs = np.linspace(brand["base_rate"].min(), brand["base_rate"].max(), 50)
ax.plot(xs, np.polyval(z, xs), "--", color="grey",
        label=f"trend (corr={corr_lift:.2f})")
ax.set(xlabel="Brand base rate (revival %)", ylabel="Top-decile lift",
       title="Lift vs base rate\n(lower base rate → higher lift)")
ax.legend(fontsize=8)

# 右：base_rate vs 每人利潤增益
ax = axes[1]
ax.scatter(brand["base_rate"], brand["gap_per_member"], s=sizes,
           c=brand["base_rate"], cmap="coolwarm_r", edgecolor="k", zorder=3)
for _, r in brand.iterrows():
    ax.annotate(r["ShopId_short"], (r["base_rate"], r["gap_per_member"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
z2 = np.polyfit(brand["base_rate"], brand["gap_per_member"], 1)
ax.plot(xs, np.polyval(z2, xs), "--", color="grey",
        label=f"trend (corr={corr_gap:.2f})")
ax.set(xlabel="Brand base rate (revival %)",
       ylabel="Model vs treat-all profit gain per member (NT$)",
       title=f"Targeting value vs base rate (uplift u={u})\n(lower base rate → higher targeting value)")
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = RA / "brand_heterogeneity.png"
plt.savefig(fig_path)
plt.close()

# ── 摘要 JSON ──
summary = {
    "decision_model": "calibrated LightGBM, time-based test",
    "profit_model": f"uplift model, u={u}, gm={gm}, contact_cost={contact_cost}",
    "n_brands": len(brand),
    "corr_baserate_lift": round(float(corr_lift), 3),
    "corr_baserate_gap_per_member": round(float(corr_gap), 3),
    "conclusion": conclusion,
    "table": brand.drop(columns=["ShopId"]).to_dict(orient="records"),
}
json.dump(summary, open(RA / "brand_heterogeneity.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"\n[完成] 輸出:")
print(f"  {csv_path}")
print(f"  {fig_path}")
print(f"  {RA / 'brand_heterogeneity.json'}")
