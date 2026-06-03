"""
Sprint B3 — 建模
================
切分：時間外驗證（早期 t0 train / 晚期 t0 test）+ 次要 stratified 80/20。
不平衡：class_weight / scale_pos_weight，不 resample（保住校準）。
模型：邏輯迴歸、LightGBM、CatBoost。
基準：多數類、RFM 規則、NAPL 規則。
校準：reliability 偏移時用 isotonic 重校。
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from catboost import CatBoostClassifier

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
OUT_DIR = ROOT / "output/B/B3"
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RNG = 42

# ================================================================
# 1. 載入特徵 + 定義欄位角色
# ================================================================
df = pd.read_parquet(FEATURES)
df["t0"] = pd.to_datetime(df["t0"])
print(f"[B3] 載入 features: {df.shape}")

# 不可當特徵：標識、標籤、洩漏欄
DROP_COLS = ["ShopMemberId", "t0", "last_purchase_date",
             "revived", "future_finish_count", "future_90d_revenue"]

CAT_COLS = ["ShopId", "RegisterSourceTypeDef", "Gender",
            "t0_channel_type", "t0_channel_detail", "t0_payment_type", "age_bucket"]
BOOL_COLS = ["is_imputed", "IsAppInstalled", "IsEnableEmail",
             "IsEnablePushNotification", "IsEnableShortMessage"]

# 數值欄 = 全部 - drop - cat - bool
ALL_FEAT = [c for c in df.columns if c not in DROP_COLS]
NUM_COLS = [c for c in ALL_FEAT if c not in CAT_COLS + BOOL_COLS]

# bool 轉 int
for c in BOOL_COLS:
    df[c] = df[c].astype(int)
# 類別欄轉字串並填 Missing
for c in CAT_COLS:
    df[c] = df[c].astype(str).fillna("Missing").replace("nan", "Missing")

y = df["revived"].values
print(f"  特徵數: {len(ALL_FEAT)} (數值 {len(NUM_COLS)}, 類別 {len(CAT_COLS)}, bool {len(BOOL_COLS)})")
print(f"  基本率: {y.mean():.4f}")

# ================================================================
# 2. 切分
# ================================================================
# 2a. 時間外驗證：t0 < 2023-11-01 → train；>= → test
time_cut = pd.Timestamp("2023-11-01")
train_mask = df["t0"] < time_cut
test_mask = ~train_mask

X = df[ALL_FEAT]
X_tr, y_tr = X[train_mask], y[train_mask]
X_te, y_te = X[test_mask], y[test_mask]
print(f"\n[切分] 時間外驗證:")
print(f"  train (t0 < 2023-11): {len(X_tr):,}  復活率 {y_tr.mean():.4f}")
print(f"  test  (t0 >= 2023-11): {len(X_te):,}  復活率 {y_te.mean():.4f}")

scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

# ================================================================
# 3. 基準模型
# ================================================================
print("\n[基準] 計算三個基準 ...")
results = {}


def evaluate(name, p_te, y_true=y_te):
    auc = roc_auc_score(y_true, p_te)
    ap = average_precision_score(y_true, p_te)
    brier = brier_score_loss(y_true, np.clip(p_te, 0, 1))
    results[name] = {"roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
                     "brier": round(brier, 4)}
    print(f"  {name:22s} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  Brier={brier:.4f}")
    return auc, ap


# 3a. 多數類（常數預測 = train 基本率）
p_majority = np.full(len(y_te), y_tr.mean())
evaluate("baseline_majority", p_majority)

# 3b. RFM-only 邏輯迴歸 baseline：只用 R/F/M 三個經典特徵訓練 logreg
#     R=hist_recency_days, F=hist_finish_count, M=hist_total_amount
#     比「人工門檻規則」更強、更公允的 baseline（讓主模型贏得更有說服力）
RFM_COLS = ["hist_recency_days", "hist_finish_count", "hist_total_amount"]
rfm_logreg = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RNG)),
])
rfm_logreg.fit(X_tr[RFM_COLS], y_tr)
p_rfm = rfm_logreg.predict_proba(X_te[RFM_COLS])[:, 1]
evaluate("baseline_rfm_logreg", p_rfm)

# 3c. NAPL 領域規則：沉睡深度 < 3× 個人週期 且 有歷史購買 → 易復活
napl_score = (
    (X_te["dormancy_gap_days"] < 3 * X_te["personal_cycle_days"]).astype(float)
    + (X_te["hist_finish_count"] > 1).astype(float)
) / 2.0
evaluate("baseline_napl_rule", napl_score.values)

# ================================================================
# 4. 邏輯迴歸（標準化 + 補值 + one-hot + class_weight）
# ================================================================
print("\n[模型] 邏輯迴歸 ...")
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                             sparse_output=False)),
])
pre = ColumnTransformer([
    ("num", num_pipe, NUM_COLS + BOOL_COLS),
    ("cat", cat_pipe, CAT_COLS),
])
logreg = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=1.0, random_state=RNG)),
])
logreg.fit(X_tr, y_tr)
p_lr = logreg.predict_proba(X_te)[:, 1]
evaluate("logreg", p_lr)
joblib.dump(logreg, MODEL_DIR / "logreg.joblib")

# ================================================================
# 5. LightGBM（原生吃 NaN + 類別欄；scale_pos_weight）
# ================================================================
print("\n[模型] LightGBM ...")
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
for c in CAT_COLS:
    X_tr_lgb[c] = X_tr_lgb[c].astype("category")
    X_te_lgb[c] = pd.Categorical(X_te_lgb[c], categories=X_tr_lgb[c].cat.categories)

lgb_clf = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
    random_state=RNG, n_jobs=-1, verbose=-1,
)
lgb_clf.fit(X_tr_lgb, y_tr, categorical_feature=CAT_COLS)
p_lgb = lgb_clf.predict_proba(X_te_lgb)[:, 1]
evaluate("lightgbm", p_lgb)
joblib.dump(lgb_clf, MODEL_DIR / "lightgbm.joblib")

# ================================================================
# 6. CatBoost（原生吃類別欄）
# ================================================================
print("\n[模型] CatBoost ...")
X_tr_cb = X_tr.copy()
X_te_cb = X_te.copy()
for c in CAT_COLS:
    X_tr_cb[c] = X_tr_cb[c].astype(str)
    X_te_cb[c] = X_te_cb[c].astype(str)
# CatBoost 不吃 NaN 於類別欄（已轉字串），數值欄可留 NaN
cb_clf = CatBoostClassifier(
    iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
    auto_class_weights="Balanced", random_seed=RNG, verbose=0,
    cat_features=CAT_COLS,
)
cb_clf.fit(X_tr_cb, y_tr)
p_cb = cb_clf.predict_proba(X_te_cb)[:, 1]
evaluate("catboost", p_cb)
cb_clf.save_model(str(MODEL_DIR / "catboost.cbm"))

# ================================================================
# 7. 校準（對主力 LightGBM 做 isotonic，於 train 內部切校準集）
# ================================================================
print("\n[校準] LightGBM isotonic ...")
# 用時間外 train 再切：前 80% fit、後 20% 校準（保時間序）
tr_sorted = df[train_mask].sort_values("t0")
cut_idx = int(len(tr_sorted) * 0.8)
fit_ids = tr_sorted.index[:cut_idx]
cal_ids = tr_sorted.index[cut_idx:]

X_fit = X.loc[fit_ids].copy()
X_cal = X.loc[cal_ids].copy()
y_fit = df.loc[fit_ids, "revived"].values
for c in CAT_COLS:
    X_fit[c] = X_fit[c].astype("category")
    X_cal[c] = pd.Categorical(X_cal[c], categories=X_fit[c].cat.categories)
    X_te_cal = X_te_lgb.copy()

lgb_base = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=63,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, scale_pos_weight=scale_pos_weight,
    random_state=RNG, n_jobs=-1, verbose=-1,
)
lgb_base.fit(X_fit, y_fit, categorical_feature=CAT_COLS)

from sklearn.isotonic import IsotonicRegression
p_cal_raw = lgb_base.predict_proba(X_cal)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_cal_raw, df.loc[cal_ids, "revived"].values)
p_lgb_cal = iso.transform(lgb_base.predict_proba(X_te_lgb)[:, 1])
evaluate("lightgbm_calibrated", p_lgb_cal)
joblib.dump({"model": lgb_base, "isotonic": iso}, MODEL_DIR / "lightgbm_calibrated.joblib")

# ================================================================
# 8. 次要對照：stratified 隨機 80/20
# ================================================================
print("\n[次要] stratified 隨機 80/20（LightGBM）...")
from sklearn.model_selection import train_test_split
Xtr2, Xte2, ytr2, yte2 = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG)
Xtr2c, Xte2c = Xtr2.copy(), Xte2.copy()
for c in CAT_COLS:
    Xtr2c[c] = Xtr2c[c].astype("category")
    Xte2c[c] = pd.Categorical(Xte2c[c], categories=Xtr2c[c].cat.categories)
spw2 = (ytr2 == 0).sum() / (ytr2 == 1).sum()
lgb2 = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    scale_pos_weight=spw2, random_state=RNG, n_jobs=-1, verbose=-1)
lgb2.fit(Xtr2c, ytr2, categorical_feature=CAT_COLS)
p2 = lgb2.predict_proba(Xte2c)[:, 1]
auc2 = roc_auc_score(yte2, p2)
ap2 = average_precision_score(yte2, p2)
results["lightgbm_random_split"] = {"roc_auc": round(auc2, 4), "pr_auc": round(ap2, 4)}
print(f"  lightgbm_random_split  ROC-AUC={auc2:.4f}  PR-AUC={ap2:.4f}")

# ================================================================
# 9. 保存預測機率 + 特徵重要度 + 結果 JSON
# ================================================================
pred_df = df.loc[test_mask, ["ShopMemberId", "ShopId", "t0", "is_imputed",
                             "revived", "future_90d_revenue"]].copy()
pred_df["p_logreg"] = p_lr
pred_df["p_lightgbm"] = p_lgb
pred_df["p_lightgbm_cal"] = p_lgb_cal
pred_df["p_catboost"] = p_cb
pred_df.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)

# LightGBM 特徵重要度
imp = pd.DataFrame({
    "feature": X_tr_lgb.columns,
    "gain": lgb_clf.booster_.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False)
imp.to_csv(OUT_DIR / "lgb_feature_importance.csv", index=False)

print("\n[特徵重要度 Top 15 — LightGBM gain]")
print(imp.head(15).to_string(index=False))

# ================================================================
# 10. 驗收：主模型是否打贏三基準
# ================================================================
main_auc = results["lightgbm"]["roc_auc"]
main_pr = results["lightgbm"]["pr_auc"]
baselines_pr = [results["baseline_majority"]["pr_auc"],
                results["baseline_rfm_logreg"]["pr_auc"],
                results["baseline_napl_rule"]["pr_auc"]]
beat_all = all(main_pr > b for b in baselines_pr)

summary = {
    "split": {
        "method": "time-based (train t0<2023-11-01, test>=)",
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "train_base_rate": round(float(y_tr.mean()), 4),
        "test_base_rate": round(float(y_te.mean()), 4),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
    },
    "n_features": len(ALL_FEAT),
    "results": results,
    "main_model": "lightgbm",
    "beats_all_baselines_pr_auc": bool(beat_all),
}
with open(OUT_DIR / "cv_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"  驗收：LightGBM PR-AUC={main_pr} vs 基準 max={max(baselines_pr)}")
print(f"  {'>>> 打贏全部三基準 <<<' if beat_all else '>>> 未全勝，需檢視 <<<'}")
print("=" * 60)
print(f"\n[B3] 完成！輸出: {MODEL_DIR}/、cv_results.json、test_predictions.parquet")
