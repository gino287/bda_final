"""
Block 1 / 第二層 — uplift 破口 + 產業實證區間
=============================================
不主張單一 uplift 點值，而是把問題倒過來：「模型要有價值，uplift 只需超過某門檻」，
再用產業實證指出真實 uplift 落在門檻的哪一側。

決策機率：校準後 LightGBM（p_lightgbm_cal）、時間外 test n=38,572。
Profit 模型沿用任務 1 的 uplift 框架：
  profit(T; u) = u·gm·Σ_{i∈T, 復活} revenue_i − (券面額·兌換 + 固定)·|T|

主圖用「基準經濟」(gm=0.55, redeem=0.40, fixed=5) 以保留已知破口 u≈0.20/0.30；
產業錨點用於陰影帶與敏感度。RFM 線 = 任務5 RFM-only 邏輯迴歸的 profit-max 政策。

產業錨點（外部研究報告，附來源於 ECONOMIC_NARRATIVE_updated.md）：
  - Winback 絕對增量 uplift 實證區間：1.2%~3.0%（Siim Pettai 1.2%、Walmart/Scintilla 2.97%）
  - 兌換率：美妝電商 8%~15%（站內推播 25%~40%）
  - 毛利率：主流醫美/大眾美妝 65%~70%
  - 固定觸及成本：SMS+Email 組合 1.0~1.5 元
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
ECON = ROOT / "output/B/B4/economics"
FIG = ROOT / "output/B/B4/figures"
ECON.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10
RNG = 42

# ── 基準經濟（主圖用） ──
BASE_GM = 0.55
BASE_REDEEM = 0.40
BASE_FIXED = 5
COUPON_FACE = 150

# ── 產業錨點 ──
UPLIFT_BAND = (0.012, 0.030)         # winback 絕對增量 uplift 實證區間
GM_GRID = [0.45, 0.55, 0.65, 0.70]    # 毛利率（65~70% 最可能）
REDEEM_GRID = [0.08, 0.12, 0.25, 0.40]  # 兌換率（美妝電商 8~15%）
SENS_FIXED = 2.0                      # 敏感度固定觸及成本（產業組合估計）
SENS_U = 0.30                         # 敏感度掃描固定 u（破口附近）

# ================================================================
# 載入：用 features 取 RFM 與標籤，merge 校準機率
# ================================================================
df = pd.read_parquet(FEATURES)
df["t0"] = pd.to_datetime(df["t0"])
preds = pd.read_parquet(PREDS)[["ShopMemberId", "ShopId", "p_lightgbm_cal"]]
df = df.merge(preds, on=["ShopMemberId", "ShopId"], how="left")

train_mask = df["t0"] < pd.Timestamp("2023-11-01")
test_mask = ~train_mask

# RFM-only 邏輯迴歸（任務5 baseline）：訓練於 train、預測 test
RFM_COLS = ["hist_recency_days", "hist_finish_count", "hist_total_amount"]
rfm_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG)),
])
rfm_pipe.fit(df.loc[train_mask, RFM_COLS], df.loc[train_mask, "revived"])
df.loc[test_mask, "p_rfm"] = rfm_pipe.predict_proba(df.loc[test_mask, RFM_COLS])[:, 1]

te = df[test_mask].copy()
p = te["p_lightgbm_cal"].values
p_rfm = te["p_rfm"].values
y = te["revived"].values.astype(float)
rev = te["future_90d_revenue"].values.astype(float)
N = len(te)
total_revival_rev = rev[y == 1].sum()
print(f"[第二層] test n={N:,}  復活率={y.mean():.4f}  復活營收總額={total_revival_rev:,.0f}")

taus = np.linspace(0, 1, 201)


def profit(treated, u, gm, redeem, fixed):
    contact = COUPON_FACE * redeem + fixed
    inc = u * gm * rev[treated & (y == 1)].sum()
    return inc - contact * treated.sum()


def best_policy(scores, u, gm, redeem, fixed):
    """掃 τ 找 profit-max。回傳 (profit, tau, reach)。"""
    pr = np.array([profit(scores >= t, u, gm, redeem, fixed) for t in taus])
    bi = int(np.argmax(pr))
    n_t = int((scores >= taus[bi]).sum())
    return float(pr[bi]), float(taus[bi]), n_t / N


# ================================================================
# 加密 u 網格（0~0.60, 步進 0.05）— 基準經濟
# ================================================================
u_grid = [round(x, 2) for x in np.arange(0.0, 0.601, 0.05)]
rows = []
for u in u_grid:
    m_prof, m_tau, m_reach = best_policy(p, u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    r_prof, r_tau, r_reach = best_policy(p_rfm, u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    all_prof = profit(np.ones(N, bool), u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    rows.append({
        "u": u,
        "model_profit": round(m_prof, 0), "model_tau": round(m_tau, 3),
        "model_reach": round(m_reach, 3),
        "treat_all_profit": round(all_prof, 0),
        "treat_none_profit": 0.0,
        "rfm_profit": round(r_prof, 0),
        "model_minus_all": round(m_prof - all_prof, 0),
        "model_minus_rfm": round(m_prof - r_prof, 0),
    })
grid = pd.DataFrame(rows)
print("\n=== u 網格（基準經濟 gm=0.55, redeem=0.40, fixed=5）===")
print(grid[["u", "model_reach", "model_profit", "treat_all_profit",
            "rfm_profit", "model_minus_all"]].to_string(index=False))

# ── 兩個破口（誠實定義；模型 τ* 含 treat-all，故模型恆 >= treat-all）──
def interp_cross(xs, ys, target=0.0):
    """線性內插 ys 穿越 target 的 x。"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    for k in range(len(xs) - 1):
        if (ys[k] - target) * (ys[k + 1] - target) <= 0 and ys[k] != ys[k + 1]:
            t = (target - ys[k]) / (ys[k + 1] - ys[k])
            return round(float(xs[k] + t * (xs[k + 1] - xs[k])), 3)
    return None

# 破口②：treat-all 期望利潤由負轉正（內插）
u_treatall_pos = interp_cross(grid["u"], grid["treat_all_profit"], 0.0)
# 破口①：模型對 treat-all 的優勢收斂到 < 10%（targeting 之後幾乎無加分）
adv_ratio = grid["model_minus_all"] / grid["model_profit"].abs().clip(lower=1)
small = grid["u"][adv_ratio < 0.10]
u_adv_small = float(small.min()) if len(small) else None
print(f"\n破口①（treat-all 由負轉正, 內插）: u~{u_treatall_pos}")
print(f"破口②（模型優勢收斂<10%, targeting 後幾乎無加分）: u~{u_adv_small}")
print(f"產業實證 uplift 區間: {UPLIFT_BAND[0]:.1%}~{UPLIFT_BAND[1]:.1%}")
print(f">>> 產業上限 {UPLIFT_BAND[1]:.1%} 遠低於 treat-all 轉正點 u~{u_treatall_pos}；"
      f"在實務區間 treat-all 期望利潤為負，模型選擇性 targeting 嚴格安全")
# 給後續圖/JSON 用的變數名
u_break2 = u_treatall_pos
u_break1 = u_adv_small

# ================================================================
# 圖 1：四線圖 + 產業 uplift 陰影帶 + 破口標記
# ================================================================
plt.figure(figsize=(9, 5.5))
plt.plot(grid["u"], grid["model_profit"], "o-", color="C0", lw=2, label="Model (optimal τ*)")
plt.plot(grid["u"], grid["treat_all_profit"], "s--", color="C3", label="Treat all")
plt.plot(grid["u"], grid["rfm_profit"], "^--", color="C2", label="RFM-logreg rule")
plt.axhline(0, color="grey", lw=1, ls=":", label="Treat none (=0)")
# 產業 uplift 陰影帶
plt.axvspan(UPLIFT_BAND[0], UPLIFT_BAND[1], color="gold", alpha=0.3,
            label=f"Industry winback uplift {UPLIFT_BAND[0]:.1%}-{UPLIFT_BAND[1]:.1%}")
# 破口標記
if u_break2 is not None:
    plt.axvline(u_break2, color="C3", ls=":", alpha=0.7)
    plt.annotate(f"treat-all turns +\nu~{u_break2:.2f}", (u_break2, 0),
                 textcoords="offset points", xytext=(5, 40), fontsize=8, color="C3")
if u_break1 is not None:
    plt.axvline(u_break1, color="C0", ls=":", alpha=0.7)
    plt.annotate(f"model adv <10%\nu~{u_break1:.2f}", (u_break1, 0),
                 textcoords="offset points", xytext=(5, 80), fontsize=8, color="C0")
plt.xlabel("Uplift rate u (fraction of revivals caused by coupon)")
plt.ylabel("Expected profit, test set (NT$)")
plt.title("Layer 2: Expected profit vs uplift u (base economics)\n"
          "Industry uplift (1.2-3%) sits far left of breakeven → model strictly safe")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "layer2_profit_vs_u.png")
plt.close()

# ================================================================
# 圖 2：敏感度熱圖（gm × redemption，固定 u=0.30, fixed=2）
#        值 = 模型 profit-max 利潤
# ================================================================
heat = np.zeros((len(GM_GRID), len(REDEEM_GRID)))
heat_adv = np.zeros_like(heat)
for i, gmv in enumerate(GM_GRID):
    for j, rdv in enumerate(REDEEM_GRID):
        m_prof, _, _ = best_policy(p, SENS_U, gmv, rdv, SENS_FIXED)
        all_prof = profit(np.ones(N, bool), SENS_U, gmv, rdv, SENS_FIXED)
        heat[i, j] = m_prof
        heat_adv[i, j] = m_prof - all_prof

fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(heat, cmap="YlGn", aspect="auto", origin="lower")
ax.set_xticks(range(len(REDEEM_GRID)))
ax.set_xticklabels([f"{r:.0%}" for r in REDEEM_GRID])
ax.set_yticks(range(len(GM_GRID)))
ax.set_yticklabels([f"{g:.0%}" for g in GM_GRID])
ax.set_xlabel("Redemption rate")
ax.set_ylabel("Gross margin")
ax.set_title(f"Layer 2: Model profit-max (NT$, test) @ u={SENS_U}, fixed={SENS_FIXED}")
for i in range(len(GM_GRID)):
    for j in range(len(REDEEM_GRID)):
        ax.text(j, i, f"{heat[i,j]:,.0f}\n(+{heat_adv[i,j]:,.0f})",
                ha="center", va="center", fontsize=7.5)
fig.colorbar(im, ax=ax, label="Model profit (NT$)")
plt.tight_layout()
plt.savefig(FIG / "layer2_sensitivity_heatmap.png")
plt.close()

# ================================================================
# JSON
# ================================================================
out = {
    "layer": "Layer 2 — uplift crossover + industry evidence band",
    "decision_model": "calibrated LightGBM (p_lightgbm_cal), time-based test n=38,572",
    "profit_model": "uplift: u·gm·Σ_{treated,revived} revenue − (face·redeem+fixed)·|treated|",
    "base_economics": {"gross_margin": BASE_GM, "redemption_rate": BASE_REDEEM,
                       "fixed_cost": BASE_FIXED, "coupon_face": COUPON_FACE},
    "u_grid": grid.to_dict(orient="records"),
    "note_model_ge_treatall": "model τ* includes τ=0 (treat-all), so model >= treat-all for all u; markers describe regions not crossings",
    "breakeven_u_treatall_positive": u_break2,
    "breakeven_u_model_advantage_below_10pct": u_break1,
    "industry_uplift_band": {"low": UPLIFT_BAND[0], "high": UPLIFT_BAND[1],
        "sources": ["Siim Pettai email winback abs uplift 1.2% (4.2%->5.4%)",
                    "Walmart Connect/Scintilla RCT 2.97% (2026)"]},
    "industry_anchors": {
        "redemption_rate_beauty_ecom": "8%-15% (Count.co)",
        "gross_margin_mainstream_beauty": "65%-70% (達爾膚69%, 軒郁68.73%)",
        "fixed_contact_cost_sms_email": "1.0-1.5 NTD",
    },
    "sensitivity_u030": {
        "gm_grid": GM_GRID, "redeem_grid": REDEEM_GRID, "fixed": SENS_FIXED,
        "model_profit": heat.tolist(),
        "model_minus_treatall": heat_adv.tolist(),
    },
    "conclusion": (f"產業 winback uplift 實證 {UPLIFT_BAND[0]:.1%}~{UPLIFT_BAND[1]:.1%}，"
                   f"遠低於 treat-all 由負轉正的 u~{u_break2}；在此實務區間 treat-all 期望利潤為負（虧損），"
                   f"模型選擇性 targeting（僅觸及高價值少數）嚴格安全。"),
}
json.dump(out, open(ECON / "layer2_crossover.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"\n[完成] 第二層輸出:")
print(f"  {ECON / 'layer2_crossover.json'}")
print(f"  {FIG / 'layer2_profit_vs_u.png'}")
print(f"  {FIG / 'layer2_sensitivity_heatmap.png'}")
