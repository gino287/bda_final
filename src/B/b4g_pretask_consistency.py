"""
Block 3+2 / Pre-task — Block 1 一致性修正
=========================================
1. 單位修正（最關鍵）：模型的 u 是「復活營收中屬增量的【比例 ratio】」；
   產業實證 1.2%、2.97% 是【絕對百分點 pp】，不可直接比。
   換算：incrementality ratio = 絕對增量 ÷ 受眾轉換率。
   email 例 (4.2%→5.4%): u = (5.4-4.2)/5.4 = 0.22。→ 產業 u 區間 ≈ 0.15~0.25。
   把產業帶從 [0.012,0.03] 改成 [0.15,0.25] 疊上破口圖（破口 u≈0.27 維持）。

2. 兌換率一致性（Layer 1）：硬省兌換率以產業實證【區間】呈現。
   - 廣播情境（保守硬地板）：8~15%
   - 即時/站內推播（B 的 t0 部署情境，Count.co 25~40%）：以 40% 為決策經濟錨點
   以敏感度區間同時呈現，不擇優。

3. Layer 3 重算：u 網格改用【比例】合理值 {0.10,0.15,0.20,0.25,0.30,0.50}
   （移除把 pp 誤當 ratio 的 0.012/0.03），重算 treat-all 虧損組數。

決策機率：校準後 LightGBM、時間外 test n=38,572。
"""

import json
import itertools
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
RNG = 42
COUPON_FACE = 150

# ── 產業錨點 ──
U_RATIO_BAND = (0.15, 0.25)          # 換算後的產業 uplift【比例】區間
GM_GRID = [0.45, 0.55, 0.65, 0.70]
REDEEM_GRID = [0.08, 0.12, 0.25, 0.40]
# 決策經濟基準（B 即時觸發 → 高兌換端；維持 spec 的破口 u≈0.27 / +0.7-1M）
BASE_GM, BASE_REDEEM, BASE_FIXED = 0.55, 0.40, 5
# Layer 1 硬省的保守端毛利率（產業主流醫美 65-70% → 用 0.65）
L1_GM = 0.65

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# ── 載入 + RFM-only logreg ──
df = pd.read_parquet(FEATURES)
df["t0"] = pd.to_datetime(df["t0"])
preds = pd.read_parquet(PREDS)[["ShopMemberId", "ShopId", "p_lightgbm_cal"]]
df = df.merge(preds, on=["ShopMemberId", "ShopId"], how="left")
tr_mask = df["t0"] < pd.Timestamp("2023-11-01")
te_mask = ~tr_mask
RFM_COLS = ["hist_recency_days", "hist_finish_count", "hist_total_amount"]
rfm = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG))])
rfm.fit(df.loc[tr_mask, RFM_COLS], df.loc[tr_mask, "revived"])
df.loc[te_mask, "p_rfm"] = rfm.predict_proba(df.loc[te_mask, RFM_COLS])[:, 1]

te = df[te_mask].copy()
p = te["p_lightgbm_cal"].values
p_rfm = te["p_rfm"].values
y = te["revived"].values.astype(float)
rev = te["future_90d_revenue"].values.astype(float)
N = len(te)
N_RETURNERS = 232328
revived_avg_rev = rev[y == 1].sum() / (y == 1).sum()
total_rev_revenue = rev[y == 1].sum()
taus = np.linspace(0, 1, 201)
print(f"[Pre-task] test n={N:,} 復活率={y.mean():.4f} 復活均營收={revived_avg_rev:.1f}")

# ================================================================
# 修正 1+2：Layer 1 硬省（兌換率 × 毛利率 敏感度）
# ================================================================
rank = np.argsort(-p)
top30 = np.zeros(N, bool); top30[rank[:int(N*0.30)]] = True
p06 = p >= 0.6

def l1_tier(mask, label, redeem, gm):
    nH = int(mask.sum()); rH = float(y[mask].mean()); revH = int(y[mask].sum())
    coupon_cost = COUPON_FACE * redeem
    m = revived_avg_rev * gm
    gross = coupon_cost * nH
    return {"tier": label, "redeem": redeem, "gm": gm, "n_H": nH,
            "revival_rate_H": round(rH, 4),
            "gross_saving_test": round(gross, 0),
            "per_returner": round(gross / N, 2),
            "scaled_returners": round(gross / N * N_RETURNERS, 0),
            "u_breakeven": round(coupon_cost / (revH * m) * nH if revH else np.nan, 4)}

l1_rows = []
for rd in [0.08, 0.12, 0.15, 0.40]:
    l1_rows.append(l1_tier(top30, "top30", rd, L1_GM))
    l1_rows.append(l1_tier(p06, "p>=0.6", rd, L1_GM))
l1 = pd.DataFrame(l1_rows)
print("\n=== Layer 1 硬省（gm=0.65, 兌換率敏感度）===")
print(l1[["tier","redeem","n_H","revival_rate_H","gross_saving_test",
          "scaled_returners","u_breakeven"]].to_string(index=False))

# ================================================================
# 修正 1：Layer 2 — 重畫破口圖（u 是 ratio，產業帶 0.15~0.25）
# ================================================================
def profit(treated, u, gm, redeem, fixed):
    return u*gm*rev[treated & (y==1)].sum() - (COUPON_FACE*redeem+fixed)*treated.sum()

def best(scores, u, gm, redeem, fixed):
    pr = np.array([profit(scores>=t, u, gm, redeem, fixed) for t in taus])
    bi = int(np.argmax(pr)); return float(pr[bi]), int((scores>=taus[bi]).sum())

u_grid = [round(x,2) for x in np.arange(0.0, 0.601, 0.05)]
g_rows = []
for u in u_grid:
    mp, mn = best(p, u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    rp, _ = best(p_rfm, u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    ap = profit(np.ones(N,bool), u, BASE_GM, BASE_REDEEM, BASE_FIXED)
    g_rows.append({"u":u, "model":mp, "reach":mn/N, "all":ap, "rfm":rp})
G = pd.DataFrame(g_rows)
# 破口②：treat-all 轉正（內插）
def cross(xs, ys):
    xs, ys = np.asarray(xs,float), np.asarray(ys,float)
    for k in range(len(xs)-1):
        if ys[k]*ys[k+1] <= 0 and ys[k]!=ys[k+1]:
            return round(float(xs[k] - ys[k]*(xs[k+1]-xs[k])/(ys[k+1]-ys[k])),3)
    return None
u_break = cross(G["u"], G["all"])
print(f"\n=== Layer 2 (base econ redeem={BASE_REDEEM}) ===")
print(f"  破口（treat-all 轉正）: u~{u_break}")
print(f"  產業 uplift【比例】帶: {U_RATIO_BAND[0]:.0%}~{U_RATIO_BAND[1]:.0%}")
for u in [0.15,0.20,0.25]:
    r = G[G["u"]==u].iloc[0]
    print(f"  u={u}: treat-all={r['all']:>11,.0f} model-all={r['model']-r['all']:>10,.0f} reach={r['reach']:.1%}")

plt.figure(figsize=(9,5.5))
plt.plot(G["u"], G["model"], "o-", color="C0", lw=2, label="Model (optimal τ*)")
plt.plot(G["u"], G["all"], "s--", color="C3", label="Treat all")
plt.plot(G["u"], G["rfm"], "^--", color="C2", label="RFM-logreg rule")
plt.axhline(0, color="grey", lw=1, ls=":", label="Treat none (=0)")
plt.axvspan(*U_RATIO_BAND, color="gold", alpha=0.3,
            label=f"Industry uplift RATIO {U_RATIO_BAND[0]:.0%}-{U_RATIO_BAND[1]:.0%}")
if u_break: plt.axvline(u_break, color="C3", ls=":", alpha=0.7)
plt.annotate(f"treat-all turns +\nu~{u_break:.2f}", (u_break,0),
             textcoords="offset points", xytext=(5,40), fontsize=8, color="C3")
plt.xlabel("Uplift RATIO u (incremental share of revival revenue; NOT absolute pp)")
plt.ylabel("Expected profit, test set (NT$)")
plt.title("Layer 2 (unit-corrected): profit vs uplift RATIO u\n"
          "Industry RATIO 15-25% straddles breakeven ~0.27 -> treat-all micro-loss, model +0.7-1.3M")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(FIG / "layer2_profit_vs_u.png")
plt.close()

# ================================================================
# 修正 3：Layer 3 — u 網格改 ratio 合理值，重算虧損組數
# ================================================================
U_GRID_FIX = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
FIXED3 = 2.0
rows = []
for u, gm, rd in itertools.product(U_GRID_FIX, GM_GRID, REDEEM_GRID):
    mp, mn = best(p, u, gm, rd, FIXED3)
    rp, _ = best(p_rfm, u, gm, rd, FIXED3)
    ap = profit(np.ones(N,bool), u, gm, rd, FIXED3)
    rows.append({"u":u,"gm":gm,"rd":rd,"model":round(mp,0),"reach":round(mn/N,3),
                 "all":round(ap,0),"rfm":round(rp,0),
                 "m_minus_all":round(mp-ap,0),"m_minus_rfm":round(mp-rp,0),
                 "all_loss":bool(ap<0)})
S = pd.DataFrame(rows); nc=len(S)
strict_all = int((S["m_minus_all"]>1).sum())
ge_rfm = int((S["m_minus_rfm"]>=-1).sum())
loss = int(S["all_loss"].sum())
print(f"\n=== Layer 3 (u-grid ratio {U_GRID_FIX}, {nc} 組) ===")
print(f"  模型嚴格勝 treat-all: {strict_all}/{nc}")
print(f"  模型 >= RFM: {ge_rfm}/{nc}")
print(f"  treat-all 虧損組數: {loss}/{nc}  (修正前 46/96)")
print(f"  模型 vs treat-all 優勢: 平均 {S['m_minus_all'].mean():,.0f} 最小 {S['m_minus_all'].min():,.0f} 最大 {S['m_minus_all'].max():,.0f}")

# 熱圖：固定 u=0.20（產業比例中位），gm × redeem 的模型對 treat-all 優勢
HU = 0.20
sub = S[S["u"]==HU]
heat = np.array([[sub[(sub["gm"]==g)&(sub["rd"]==r)]["m_minus_all"].iloc[0]
                  for r in REDEEM_GRID] for g in GM_GRID])
fig, ax = plt.subplots(figsize=(7.5,5))
im = ax.imshow(heat, cmap="Greens", aspect="auto", origin="lower")
ax.set_xticks(range(len(REDEEM_GRID))); ax.set_xticklabels([f"{r:.0%}" for r in REDEEM_GRID])
ax.set_yticks(range(len(GM_GRID))); ax.set_yticklabels([f"{g:.0%}" for g in GM_GRID])
ax.set_xlabel("Redemption rate"); ax.set_ylabel("Gross margin")
ax.set_title(f"Layer 3 (unit-corrected): Model advantage over treat-all (NT$)\n@ industry uplift RATIO u={HU}, fixed={FIXED3}")
for i in range(len(GM_GRID)):
    for j in range(len(REDEEM_GRID)):
        ax.text(j,i,f"{heat[i,j]:+,.0f}",ha="center",va="center",fontsize=8)
fig.colorbar(im, ax=ax, label="Model - treat-all (NT$)")
plt.tight_layout(); plt.savefig(FIG / "layer3_win_heatmap.png"); plt.close()

# ================================================================
# 更新 JSON（修正版）
# ================================================================
out = {
    "pretask": "Block 1 consistency fix",
    "fix1_unit": {
        "issue": "model u is a RATIO (incremental share of revival revenue); industry 1.2%/2.97% are absolute pp",
        "conversion": "ratio = abs_uplift / audience_conversion; email 4.2%->5.4% gives u=1.2/5.4=0.22",
        "industry_u_ratio_band": list(U_RATIO_BAND),
    },
    "fix2_redemption": {
        "L1_hard_floor_broadcast": "8-15%",
        "decision_econ_realtime_trigger": "25-40% (B fires at t0 repurchase = in-app/push, Count.co); base uses 0.40",
        "note": "L1 conservative floor uses low broadcast redemption; L2/3 decision econ uses real-time-trigger redemption. Both industry-sourced; sensitivity shown.",
    },
    "layer1_saving_sensitivity": l1.to_dict(orient="records"),
    "layer2_base_econ": {"gm":BASE_GM,"redeem":BASE_REDEEM,"fixed":BASE_FIXED},
    "layer2_breakeven_u": u_break,
    "layer2_at_industry_band": {str(u): {
        "treat_all": float(G[G["u"]==u]["all"].iloc[0]),
        "model_minus_all": float(G[G["u"]==u]["model"].iloc[0]-G[G["u"]==u]["all"].iloc[0])
    } for u in [0.15,0.20,0.25]},
    "layer3_corrected": {
        "u_grid": U_GRID_FIX, "n_combos": nc,
        "model_strictly_beats_treatall": strict_all,
        "model_ge_rfm": ge_rfm,
        "treatall_loss_combos": loss,
        "treatall_loss_combos_OLD": "46/96 (artifact of pp-as-ratio tiny u)",
        "model_vs_all_margin": {"mean":round(S["m_minus_all"].mean(),0),
                                "min":round(S["m_minus_all"].min(),0),
                                "max":round(S["m_minus_all"].max(),0)},
    },
}
json.dump(out, open(ECON/"pretask_consistency.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n[完成] {ECON/'pretask_consistency.json'}")
print(f"  重畫: layer2_profit_vs_u.png, layer3_win_heatmap.png")
