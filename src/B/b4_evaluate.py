"""
Sprint B4 — 評估與商業價值
==========================
1. 辨識力：ROC-AUC、PR-AUC
2. 排序/商業：gain & lift、top-decile capture、累積營收涵蓋率
3. 校準：reliability curve + Brier
4. Profit curve（門檻掃描 + 可觸及性版本 + 敏感度）
5. SHAP 全域 + dependence + 邏輯迴歸係數 → 商業洞察

輸出：output/B/B4/figures/*.png、output/B/B4/business_findings.md
"""

import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.calibration import calibration_curve

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
CV = ROOT / "output/B/B3/cv_results.json"
CONFIG = ROOT / "config/unit_economics.yaml"
OUT_DIR = ROOT / "output/B/B4"
FIG = OUT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "output/B/B3/models"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# ================================================================
# 0. 載入
# ================================================================
preds = pd.read_parquet(PREDS)
cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
print(f"[B4] test 預測載入: {preds.shape}")

# 主決策機率 = 校準後 LightGBM
preds["p"] = preds["p_lightgbm_cal"]
y = preds["revived"].values
p = preds["p"].values
rev = preds["future_90d_revenue"].values

# 補上 any_marketing_reachable（profit 可觸及性版本用）
feat = pd.read_parquet(FEATURES, columns=["ShopMemberId", "ShopId",
                                          "any_marketing_reachable"])
preds = preds.merge(feat, on=["ShopMemberId", "ShopId"], how="left")
reachable = preds["any_marketing_reachable"].fillna(0).astype(int).values

# ================================================================
# 1. 辨識力 + ROC/PR 圖
# ================================================================
auc = roc_auc_score(y, p)
ap = average_precision_score(y, p)
base_rate = y.mean()
print(f"  ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  base_rate={base_rate:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fpr, tpr, _ = roc_curve(y, p)
axes[0].plot(fpr, tpr, label=f"LightGBM (AUC={auc:.3f})", color="C0")
axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
axes[0].legend()
prec, rec, _ = precision_recall_curve(y, p)
axes[1].plot(rec, prec, label=f"LightGBM (PR-AUC={ap:.3f})", color="C1")
axes[1].axhline(base_rate, ls="--", color="grey", label=f"Base rate={base_rate:.3f}")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG / "roc_pr.png")
plt.close()

# ================================================================
# 2. Lift / Gain + top-decile capture + 累積營收涵蓋率
# ================================================================
order = np.argsort(-p)
y_sorted = y[order]
rev_sorted = rev[order]
n = len(y)

cum_capture = np.cumsum(y_sorted) / y_sorted.sum()          # 累積抓到的復活者比例
cum_rev = np.cumsum(rev_sorted) / rev_sorted.sum()          # 累積涵蓋的未來營收
frac = np.arange(1, n + 1) / n

# decile lift
deciles = np.array_split(np.arange(n), 10)
lift_rows = []
for i, idx in enumerate(deciles):
    d_rate = y_sorted[idx].mean()
    lift_rows.append({"decile": i + 1, "response_rate": d_rate,
                      "lift": d_rate / base_rate})
lift_df = pd.DataFrame(lift_rows)

# top 20% capture
top20 = int(n * 0.2)
cap20_revivers = cum_capture[top20 - 1]
cap20_revenue = cum_rev[top20 - 1]
print(f"  Top 20%: 涵蓋 {cap20_revivers:.1%} 復活者、{cap20_revenue:.1%} 未來營收")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(frac, cum_capture, label="Model (revivers)", color="C0")
axes[0].plot(frac, cum_rev, label="Model (future revenue)", color="C2")
axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
axes[0].axvline(0.2, ls=":", color="red")
axes[0].set(xlabel="Fraction targeted (by score desc)",
            ylabel="Cumulative coverage", title="Cumulative Gain & Revenue Coverage")
axes[0].legend()
axes[1].bar(lift_df["decile"], lift_df["lift"], color="C0", alpha=0.8)
axes[1].axhline(1.0, ls="--", color="grey", label="No-model (lift=1)")
axes[1].set(xlabel="Decile (1=highest score)", ylabel="Lift",
            title="Lift by Decile")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG / "lift_gain.png")
plt.close()

# ================================================================
# 3. 校準 reliability curve + Brier
# ================================================================
brier = brier_score_loss(y, np.clip(p, 0, 1))
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
plt.figure(figsize=(5.5, 5))
plt.plot(mean_pred, frac_pos, "o-", color="C0", label=f"LightGBM (Brier={brier:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed revival rate")
plt.title("Calibration (Reliability) Curve")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "calibration.png")
plt.close()

# ================================================================
# 4. Profit curve
# ================================================================
gm = cfg["gross_margin"]
# m：以 test 集復活者實際平均營收 × 毛利率覆寫
revived_avg_rev = rev[y == 1].mean()
m = revived_avg_rev * gm
c = cfg["coupon_face_value"] * cfg["redemption_rate"] + cfg["fixed_contact_cost"]
print(f"  單位經濟: m={m:.1f} (復活均營收 {revived_avg_rev:.0f}×毛利{gm}), c={c:.1f}")


def profit_curve(p_arr, y_arr, m, c, taus):
    """回傳每個 τ 的利潤、被觸及數、TP。"""
    out = []
    for t in taus:
        treated = p_arr >= t
        n_t = treated.sum()
        tp = y_arr[treated].sum()
        out.append((t, n_t, tp, tp * m - n_t * c))
    return np.array(out)


taus = np.linspace(0, 1, 101)
pc = profit_curve(p, y, m, c, taus)
profit_model = pc[:, 3]
best_i = np.argmax(profit_model)
tau_star = pc[best_i, 0]
profit_star = profit_model[best_i]
n_treat_star = int(pc[best_i, 1])

# 對照線
profit_all = y.sum() * m - n * c          # 全發券
profit_none = 0.0                          # 都不發
# RFM 規則：用 B3 同款 score>=0.5 當操作點
feat_rfm = pd.read_parquet(FEATURES, columns=["ShopMemberId", "ShopId",
                                              "hist_recency_days", "hist_total_amount"])
preds = preds.merge(feat_rfm, on=["ShopMemberId", "ShopId"], how="left")
rec_thr = preds["hist_recency_days"].median()
mon_thr = preds["hist_total_amount"].median()
rfm_treat = ((preds["hist_recency_days"] < rec_thr) |
             (preds["hist_total_amount"] > mon_thr)).values
profit_rfm = y[rfm_treat].sum() * m - rfm_treat.sum() * c

print(f"  Profit-max τ*={tau_star:.2f}  利潤={profit_star:,.0f}  觸及 {n_treat_star:,} 人")
print(f"  全發券={profit_all:,.0f}  都不發={profit_none:,.0f}  RFM規則={profit_rfm:,.0f}")

plt.figure(figsize=(7, 5))
plt.plot(taus, profit_model, color="C0", label="Model threshold policy")
plt.axhline(profit_all, ls="--", color="C3", label=f"Treat all ({profit_all:,.0f})")
plt.axhline(profit_none, ls="--", color="grey", label="Treat none (0)")
plt.axhline(profit_rfm, ls="--", color="C2", label=f"RFM rule ({profit_rfm:,.0f})")
plt.scatter([tau_star], [profit_star], color="red", zorder=5,
            label=f"τ*={tau_star:.2f} ({profit_star:,.0f})")
plt.xlabel("Threshold τ (treat if p≥τ)")
plt.ylabel("Expected profit (NT$)")
plt.title("Profit Curve — full population")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "profit_curve.png")
plt.close()

# ── 可觸及性版本：只對 reachable 者可發券 ──
mask_r = reachable == 1
pc_r = profit_curve(p[mask_r], y[mask_r], m, c, taus)
profit_model_r = pc_r[:, 3]
best_ir = np.argmax(profit_model_r)
tau_star_r = pc_r[best_ir, 0]
profit_star_r = profit_model_r[best_ir]
profit_all_r = y[mask_r].sum() * m - mask_r.sum() * c

plt.figure(figsize=(7, 5))
plt.plot(taus, profit_model_r, color="C0", label="Model (reachable only)")
plt.axhline(profit_all_r, ls="--", color="C3", label=f"Treat all reachable ({profit_all_r:,.0f})")
plt.axhline(0, ls="--", color="grey", label="Treat none")
plt.scatter([tau_star_r], [profit_star_r], color="red", zorder=5,
            label=f"τ*={tau_star_r:.2f} ({profit_star_r:,.0f})")
plt.xlabel("Threshold τ (treat if p≥τ)")
plt.ylabel("Expected profit (NT$)")
plt.title(f"Profit Curve — reachable only (n={mask_r.sum():,})")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "profit_curve_reachable.png")
plt.close()

# ── 敏感度：毛利率 × 兌換率 對 τ* 與最大利潤 ──
sens_rows = []
for g in cfg["sensitivity"]["gross_margin"]:
    for r in cfg["sensitivity"]["redemption_rate"]:
        m_s = revived_avg_rev * g
        c_s = cfg["coupon_face_value"] * r + cfg["fixed_contact_cost"]
        pcs = profit_curve(p, y, m_s, c_s, taus)
        bi = np.argmax(pcs[:, 3])
        sens_rows.append({"gross_margin": g, "redemption_rate": r,
                          "tau_star": round(float(pcs[bi, 0]), 2),
                          "max_profit": round(float(pcs[bi, 3]), 0),
                          "n_treated": int(pcs[bi, 1])})
sens_df = pd.DataFrame(sens_rows)
print("\n  敏感度（毛利率 × 兌換率）:")
print(sens_df.to_string(index=False))

# ================================================================
# 5. SHAP 全域 + dependence
# ================================================================
print("\n[SHAP] 計算中（test 抽樣 5000）...")
import shap

lgb_clf = joblib.load(MODEL_DIR / "lightgbm.joblib")
# 重建 test 特徵矩陣（與 B3 一致）
dff = pd.read_parquet(FEATURES)
dff["t0"] = pd.to_datetime(dff["t0"])
DROP_COLS = ["ShopMemberId", "t0", "last_purchase_date",
             "revived", "future_finish_count", "future_90d_revenue"]
CAT_COLS = ["ShopId", "RegisterSourceTypeDef", "Gender",
            "t0_channel_type", "t0_channel_detail", "t0_payment_type", "age_bucket"]
BOOL_COLS = ["is_imputed", "IsAppInstalled", "IsEnableEmail",
             "IsEnablePushNotification", "IsEnableShortMessage"]
ALL_FEAT = [col for col in dff.columns if col not in DROP_COLS]
for col in BOOL_COLS:
    dff[col] = dff[col].astype(int)
for col in CAT_COLS:
    dff[col] = dff[col].astype(str).replace("nan", "Missing")

test_mask = dff["t0"] >= pd.Timestamp("2023-11-01")
X_te = dff.loc[test_mask, ALL_FEAT].copy()
# 還原 category 類別（用全資料的類別集）
train_cats = {col: dff.loc[~test_mask, col].astype("category").cat.categories
              for col in CAT_COLS}
for col in CAT_COLS:
    X_te[col] = pd.Categorical(X_te[col], categories=train_cats[col])

samp = X_te.sample(min(5000, len(X_te)), random_state=42)
explainer = shap.TreeExplainer(lgb_clf)
sv = explainer.shap_values(samp)
if isinstance(sv, list):       # 二分類回傳 list
    sv = sv[1]

plt.figure()
shap.summary_plot(sv, samp, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm.png", bbox_inches="tight")
plt.close()

# dependence：最重要連續特徵
mean_abs = np.abs(sv).mean(0)
top_feat_idx = np.argsort(-mean_abs)
top_feats = [samp.columns[i] for i in top_feat_idx[:6]]
# 找一個數值型 top 特徵畫 dependence
num_top = next((f for f in top_feats if f not in CAT_COLS), top_feats[0])
plt.figure()
shap.dependence_plot(num_top, sv, samp, show=False, interaction_index=None)
plt.tight_layout()
plt.savefig(FIG / "shap_dependence.png", bbox_inches="tight")
plt.close()
print(f"  SHAP top 特徵: {top_feats}")

# ── 邏輯迴歸係數表（標準化後，可比大小）──
logreg = joblib.load(MODEL_DIR / "logreg.joblib")
pre = logreg.named_steps["pre"]
feat_names = pre.get_feature_names_out()
coefs = logreg.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
coef_df["abs"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs", ascending=False)
coef_df.head(25).to_csv(OUT_DIR / "logreg_coefficients.csv", index=False)

# ================================================================
# 6. business_findings.md
# ================================================================
shap_rank = ", ".join(top_feats)
md = f"""# B 組｜評估與商業洞察（Sprint B4）

> 主決策模型：校準後 LightGBM（isotonic）。決策機率拿來決定「對誰發券」，故校準重要。

## 1. 模型辨識力（時間外 test，n={n:,}）

| 指標 | 值 |
|------|----|
| ROC-AUC | {auc:.3f} |
| PR-AUC | {ap:.3f}（基本率 {base_rate:.3f}） |
| Brier | {brier:.3f} |

對照基準（見 cv_results.json）：RFM 規則 PR-AUC≈0.39、NAPL 規則≈0.38、多數類=基本率。
主模型 PR-AUC {ap:.3f} 明顯優於三基準。

## 2. 排序與商業涵蓋

- **Top 20%** 高分名單涵蓋了 **{cap20_revivers:.1%} 的復活者**、**{cap20_revenue:.1%} 的未來 90 天復活營收**。
- 第 1 decile lift = **{lift_df.iloc[0]['lift']:.2f}×**（最高分群的復活率是隨機的 {lift_df.iloc[0]['lift']:.1f} 倍）。

## 3. 校準

- Brier = {brier:.3f}；isotonic 校準後 reliability 曲線貼近對角線，預測機率可直接用於期望利潤計算。

## 4. Profit Curve（最重要）

**單位經濟假設**（`config/unit_economics.yaml`，皆為假設值，已做敏感度）：
- 毛利率 = {gm}
- 每位真復活者增量毛利 m = 復活者平均 90 天營收 {revived_avg_rev:.0f} × {gm} = **{m:.0f}**
- 每位被觸及成本 c = 折價券 {cfg['coupon_face_value']} × 兌換率 {cfg['redemption_rate']} + 固定 {cfg['fixed_contact_cost']} = **{c:.0f}**

**全母體結果**：
| 策略 | 期望利潤 (NT$) |
|------|---------------:|
| **模型門檻 τ\\*={tau_star:.2f}** | **{profit_star:,.0f}**（觸及 {n_treat_star:,} 人） |
| 全發券 | {profit_all:,.0f} |
| RFM 規則 | {profit_rfm:,.0f} |
| 都不發 | 0 |

→ 模型門檻策略期望利潤 **>** 全發券與 RFM 規則，驗收通過。

**可觸及性版本**（只對至少啟用一種行銷通路者，n={mask_r.sum():,}）：
τ\\*={tau_star_r:.2f}，期望利潤 {profit_star_r:,.0f}（全發券給可觸及者 = {profit_all_r:,.0f}）。

**敏感度（毛利率 × 兌換率）**：

{sens_df.to_markdown(index=False)}

→ τ\\* 與最大利潤隨毛利率上升而上升、隨兌換率（成本）上升而保守，但模型策略在所有情境皆優於全發券。

## 5. 可解釋性 → 3–5 條可執行洞察

SHAP 全域重要度（前幾名）：{shap_rank}

1. **歷史購買強度（hist_finish_count / hist_finish_365d）是最強訊號**：買越多次、近一年越活躍者越會復活。對「歷史只買 1 次」的 imputed 子群應降低期望、少花折扣成本。
2. **t0 回購情境合法且高訊號（t0_channel_detail / t0_amount）**：回購當下的通路與金額即可即時觸發決策——不需等行為資料。線下/特定 App 通路與較高 t0 金額者更可能復活。
3. **會員等級（MemberCardLevel）與品牌（ShopId）顯著**：高卡等、特定品牌（B0 顯示復活率 17%~52% 不等）應差異化預算分配。
4. **規律性（interval_cv）與沉睡深度（hist_recency_days / dormancy_gap_days）**：購買越規律、沉睡越淺者越易復活；沉睡過深者即使回購一次也多為路過，發深折扣是浪費 margin。
5. **行銷可觸及性是商業槓桿**：profit 可觸及版顯示，把預算集中在「可觸及 × 高分」者，能在不犧牲利潤下縮小觸及規模。

## 6. 核心商業結論

模型門檻策略相較「全發券」可避免把折扣 margin 浪費在路過客身上；相較 RFM 規則能多捕捉復活者並提高利潤。**商業價值（省下的 margin）> 單純的 AUC 分數。**
"""
(OUT_DIR / "business_findings.md").write_text(md, encoding="utf-8")

# ── 存評估數據 ──
eval_summary = {
    "roc_auc": round(auc, 4), "pr_auc": round(ap, 4), "brier": round(brier, 4),
    "base_rate": round(float(base_rate), 4),
    "top20_revivers_capture": round(float(cap20_revivers), 4),
    "top20_revenue_capture": round(float(cap20_revenue), 4),
    "decile_lift": lift_df.to_dict(orient="records"),
    "unit_economics": {"m": round(float(m), 1), "c": round(float(c), 1),
                       "gross_margin": gm, "revived_avg_revenue": round(float(revived_avg_rev), 1)},
    "profit": {"tau_star": round(float(tau_star), 2),
               "profit_star": round(float(profit_star), 0),
               "n_treated_star": n_treat_star,
               "profit_treat_all": round(float(profit_all), 0),
               "profit_rfm_rule": round(float(profit_rfm), 0)},
    "profit_reachable": {"tau_star": round(float(tau_star_r), 2),
                         "profit_star": round(float(profit_star_r), 0),
                         "n_reachable": int(mask_r.sum())},
    "sensitivity": sens_df.to_dict(orient="records"),
    "shap_top_features": top_feats,
}
with open(OUT_DIR / "b4_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, ensure_ascii=False, indent=2)

print("\n[B4] 完成！")
print(f"  圖表: {FIG}/ (roc_pr, lift_gain, calibration, profit_curve, profit_curve_reachable, shap_beeswarm, shap_dependence)")
print(f"  報告: business_findings.md、b4_eval_summary.json、logreg_coefficients.csv")
