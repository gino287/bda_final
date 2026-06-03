"""
任務 1 — 修正版 profit 分析（uplift + cannibalization）
=====================================================
在 B4 原版 profit 之外，新增兩個更誠實的成本模型（原版保留對照）。
資料：時間外 test 集、校準後 LightGBM 機率（p_lightgbm_cal）。

原版假設：被觸及的復活者全都貢獻全額增量毛利 m → 過於樂觀（m/c≈11 → 幾乎全發）。

【模型 B — Uplift 參數化】
  只有 u 比例的復活營收是「優惠券促成的增量」。
  profit_B(T; u) = u · gm · Σ_{i∈T, 復活} revenue_i  −  (coupon·redeem + fixed) · |T|
  掃 u ∈ {0.05,0.1,0.2,0.3,0.5}，每個 u 下比較 模型τ* / 全發券 / RFM。
  結論：u 低於多少時，最佳策略從「全發券」變成「targeting（τ*>0）」。

【模型 C — Margin cannibalization】
  被觸及且「本來就會復活」者（占復活者 (1−u_c)），白送的折扣 margin 被扣掉。
  profit_C(T) = u_c · gm · Σ_{T,復活} revenue_i        （增量毛利）
              − (1−u_c) · (coupon·redeem) · |R_T|       （白送折扣的 cannibalization）
              − fixed · |T|                              （觸及成本）
  重畫 profit curve（掃 τ），找新 τ*，對照 全發券 / RFM / 都不發。
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
FEATURES = ROOT / "output/B/B2/features.parquet"
CONFIG = ROOT / "config/unit_economics.yaml"
FIG = ROOT / "output/B/B4/figures"
OUT_JSON = ROOT / "output/B/B4/b_profit_revised.json"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# ── 載入 ──
cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
rp = cfg["revised_profit"]
gm = cfg["gross_margin"]
coupon = cfg["coupon_face_value"]
redeem = cfg["redemption_rate"]
fixed = cfg["fixed_contact_cost"]
coupon_cost = coupon * redeem          # 折扣的 margin 成本（兌換才付）
contact_cost = coupon_cost + fixed     # 原版/模型B：每位被觸及總成本

preds = pd.read_parquet(PREDS)
p = preds["p_lightgbm_cal"].values
y = preds["revived"].values.astype(float)
rev = preds["future_90d_revenue"].values.astype(float)
N = len(y)

# RFM 規則操作點（與 B4 一致）
rfm = pd.read_parquet(FEATURES, columns=["ShopMemberId", "ShopId",
                                         "hist_recency_days", "hist_total_amount"])
preds = preds.merge(rfm, on=["ShopMemberId", "ShopId"], how="left")
rec_thr = preds["hist_recency_days"].median()
mon_thr = preds["hist_total_amount"].median()
rfm_treat = ((preds["hist_recency_days"] < rec_thr) |
             (preds["hist_total_amount"] > mon_thr)).values

print(f"[修正profit] test n={N:,}  復活率={y.mean():.4f}")
print(f"  單位經濟: gm={gm}, coupon_cost={coupon_cost:.1f}, fixed={fixed}, "
      f"contact_cost(模型B)={contact_cost:.1f}")
print(f"  復活者平均營收={rev[y==1].mean():.1f}  全 test 復活營收總額={rev[y==1].sum():,.0f}")

taus = np.linspace(0, 1, 201)

# ================================================================
# 模型 B — Uplift 參數化
# ================================================================
print("\n=== 模型 B：Uplift 參數化 ===")


def profit_B(treated, u):
    """treated: bool array。回傳期望利潤。"""
    n_t = treated.sum()
    inc_benefit = u * gm * rev[treated & (y == 1)].sum()
    return inc_benefit - contact_cost * n_t


uplift_grid = rp["uplift_grid"]
B_rows = []
for u in uplift_grid:
    # 模型：掃 τ 找最佳
    profits = np.array([profit_B(p >= t, u) for t in taus])
    bi = int(np.argmax(profits))
    tau_star = float(taus[bi])
    model_profit = float(profits[bi])
    n_treat = int((p >= tau_star).sum())
    # 對照
    all_profit = profit_B(np.ones(N, bool), u)
    none_profit = 0.0
    rfm_profit = profit_B(rfm_treat, u)
    # 因為 treat-all 是 τ=0、已含在掃描內，模型 τ* 必 >= treat-all。
    # 有意義的「targeting」= 模型選擇性觸及（reach 明顯 < 全體）且全發券非最佳。
    selective = (n_treat / N) < 0.80
    all_is_loss = all_profit <= 0
    B_rows.append({
        "u": u, "tau_star": round(tau_star, 3),
        "reach_ratio": round(n_treat / N, 3), "n_treated": n_treat,
        "model_profit": round(model_profit, 0),
        "treat_all_profit": round(all_profit, 0),
        "rfm_profit": round(rfm_profit, 0),
        "model_vs_all_gap": round(model_profit - all_profit, 0),
        "selective_targeting": bool(selective),
        "treat_all_is_loss": bool(all_is_loss),
    })
    tag = "SELECTIVE" if selective else "~treat-all"
    print(f"  u={u:.2f}: tau*={tau_star:.3f} reach={n_treat/N:5.1%} "
          f"model={model_profit:>12,.0f} all={all_profit:>12,.0f} "
          f"gap={model_profit-all_profit:>11,.0f} {tag}")

B_df = pd.DataFrame(B_rows)

# 結論：u 低於多少時 targeting 變得有選擇性（reach<80%），且全發券開始虧損
sel_us = B_df.loc[B_df["selective_targeting"], "u"].tolist()
nonsel_us = B_df.loc[~B_df["selective_targeting"], "u"].tolist()
loss_us = B_df.loc[B_df["treat_all_is_loss"], "u"].tolist()
if sel_us and nonsel_us:
    threshold_u = max(sel_us)
    crossover = (f"u <= {threshold_u:.2f} 時模型轉為選擇性 targeting（觸及<80%）並大幅勝過全發券；"
                 f"u >= {min(nonsel_us):.2f} 時 uplift 夠高，最佳策略趨近全發券")
elif sel_us:
    threshold_u = max(sel_us)
    crossover = f"在掃描範圍內（u <= {threshold_u:.2f}）模型皆選擇性 targeting，勝過全發券"
else:
    threshold_u = None
    crossover = "在掃描範圍內 uplift 皆夠高，最佳策略趨近全發券（targeting 價值有限）"
if loss_us:
    crossover += f"。且 u <= {max(loss_us):.2f} 時全發券期望利潤為負（觸及成本吃掉增量）"
print(f"  >>> 結論：{crossover}")

# 圖：期望利潤 vs u（三條線）
plt.figure(figsize=(7.5, 5))
plt.plot(B_df["u"], B_df["model_profit"], "o-", color="C0", label="Model (optimal τ*)")
plt.plot(B_df["u"], B_df["treat_all_profit"], "s--", color="C3", label="Treat all")
plt.plot(B_df["u"], B_df["rfm_profit"], "^--", color="C2", label="RFM rule")
plt.axhline(0, color="grey", lw=0.8, ls=":")
# 標記模型選擇性 targeting 的點（reach<80%）
for _, r in B_df.iterrows():
    if r["selective_targeting"]:
        plt.annotate(f"reach={r['reach_ratio']:.0%}", (r["u"], r["model_profit"]),
                     textcoords="offset points", xytext=(0, 9),
                     fontsize=8, color="C0", ha="center")
plt.xlabel("Uplift rate u (fraction of revivals caused by coupon)")
plt.ylabel("Expected profit (NT$)")
plt.title("Revised Profit (Uplift model): Expected profit vs u")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "profit_uplift_vs_u.png")
plt.close()

# ================================================================
# 模型 C — Margin cannibalization
# ================================================================
print("\n=== 模型 C：Margin cannibalization ===")
u_c = rp["cannibalization_uplift"]


def profit_C(treated):
    n_t = treated.sum()
    rev_mask = treated & (y == 1)
    n_rev = rev_mask.sum()
    inc_benefit = u_c * gm * rev[rev_mask].sum()              # 增量毛利
    cannib = (1 - u_c) * coupon_cost * n_rev                  # 白送折扣（本來就會復活者）
    return inc_benefit - cannib - fixed * n_t


profits_C = np.array([profit_C(p >= t) for t in taus])
biC = int(np.argmax(profits_C))
tau_star_C = float(taus[biC])
model_profit_C = float(profits_C[biC])
n_treat_C = int((p >= tau_star_C).sum())
all_profit_C = profit_C(np.ones(N, bool))
none_profit_C = 0.0
rfm_profit_C = profit_C(rfm_treat)

print(f"  u_c={u_c}: τ*={tau_star_C:.3f} reach={n_treat_C/N:.1%} ({n_treat_C:,}人)")
print(f"  model={model_profit_C:,.0f}  all={all_profit_C:,.0f}  "
      f"rfm={rfm_profit_C:,.0f}  none=0")
print(f"  model vs all gap = {model_profit_C-all_profit_C:,.0f}")

plt.figure(figsize=(7.5, 5))
plt.plot(taus, profits_C, color="C0", label="Model threshold policy")
plt.axhline(all_profit_C, ls="--", color="C3", label=f"Treat all ({all_profit_C:,.0f})")
plt.axhline(none_profit_C, ls="--", color="grey", label="Treat none (0)")
plt.axhline(rfm_profit_C, ls="--", color="C2", label=f"RFM rule ({rfm_profit_C:,.0f})")
plt.scatter([tau_star_C], [model_profit_C], color="red", zorder=5,
            label=f"τ*={tau_star_C:.3f} ({model_profit_C:,.0f})")
plt.xlabel("Threshold τ (treat if p≥τ)")
plt.ylabel("Expected profit (NT$)")
plt.title(f"Revised Profit (Cannibalization model, u_c={u_c})")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "profit_cannibalization.png")
plt.close()

# ================================================================
# 輸出 JSON
# ================================================================
out = {
    "decision_model": "calibrated LightGBM (p_lightgbm_cal), time-based test",
    "test_n": int(N),
    "test_base_rate": round(float(y.mean()), 4),
    "assumptions": {
        "gross_margin": gm, "coupon_face_value": coupon,
        "redemption_rate": redeem, "fixed_contact_cost": fixed,
        "coupon_cost(=face*redeem)": round(coupon_cost, 2),
        "contact_cost_modelB(=coupon_cost+fixed)": round(contact_cost, 2),
    },
    "model_B_uplift": {
        "description": "Only u of revival revenue is incremental; cost on all treated.",
        "grid": B_df.to_dict(orient="records"),
        "crossover_conclusion": crossover,
        "targeting_threshold_u": threshold_u,
    },
    "model_C_cannibalization": {
        "description": "Subtract wasted discount margin on would-revive-anyway customers.",
        "assumed_uplift_u_c": u_c,
        "tau_star": round(tau_star_C, 3),
        "reach_ratio": round(n_treat_C / N, 3),
        "n_treated": n_treat_C,
        "model_profit": round(model_profit_C, 0),
        "treat_all_profit": round(all_profit_C, 0),
        "rfm_profit": round(rfm_profit_C, 0),
        "model_vs_all_gap": round(model_profit_C - all_profit_C, 0),
    },
}
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ================================================================
# 終端摘要報告
# ================================================================
print("\n" + "=" * 64)
print("  修正版 profit 摘要（每版本：profit-max 觸及比例 + 模型 vs 全發券差距）")
print("=" * 64)
print("\n[模型 B — Uplift 參數化] 各 u 的 profit-max 觸及比例與差距：")
print(B_df[["u", "tau_star", "reach_ratio", "model_profit",
            "treat_all_profit", "model_vs_all_gap", "selective_targeting"]]
      .to_string(index=False))
print(f"\n  結論：{crossover}")
print(f"\n[模型 C — Cannibalization, u_c={u_c}]")
print(f"  profit-max τ*={tau_star_C:.3f}，觸及比例={n_treat_C/N:.1%}（{n_treat_C:,}/{N:,}）")
print(f"  模型利潤={model_profit_C:,.0f}  vs 全發券={all_profit_C:,.0f}  "
      f"→ 差距={model_profit_C-all_profit_C:,.0f}")
print(f"\n圖：figures/profit_uplift_vs_u.png、figures/profit_cannibalization.png")
print(f"摘要：{OUT_JSON}")
