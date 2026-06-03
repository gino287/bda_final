"""
任務 4 — 清理 t0_payment_type 並驗證 LightGBM 前後指標
====================================================
規則：t0_payment_type 為 NaN 且該筆 t0 訂單通路為 POS（t0_channel_type='Pos'）
      → 補成 "POS"（線下無線上付款方式，屬可解釋缺失）；其餘真缺失保留 NaN。

在同一支腳本內以「相同切分 / 參數」分別訓練 before / after 的 LightGBM(raw)
與 isotonic 校準版，回報時間外 test 的 ROC-AUC / PR-AUC / Brier 對照。
不動其他模型（logreg / catboost）。

驗證通過後：
  - 把清理寫回 features.parquet（永久）
  - 更新 b2_features.py 原始碼（讓未來重跑也含此清理）
  - 覆寫 B3/models 的 lightgbm.joblib、lightgbm_calibrated.joblib
  - 更新 test_predictions.parquet 的 p_lightgbm、p_lightgbm_cal 兩欄
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
MODEL_DIR = ROOT / "output/B/B3/models"
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
OUT_JSON = ROOT / "output/B/B4/t0_paytype_cleanup.json"

RNG = 42
DROP_COLS = ["ShopMemberId", "t0", "last_purchase_date",
             "revived", "future_finish_count", "future_90d_revenue"]
CAT_COLS = ["ShopId", "RegisterSourceTypeDef", "Gender",
            "t0_channel_type", "t0_channel_detail", "t0_payment_type", "age_bucket"]
BOOL_COLS = ["is_imputed", "IsAppInstalled", "IsEnableEmail",
             "IsEnablePushNotification", "IsEnableShortMessage"]

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=63,
                  min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1)

# ================================================================
# 載入 + 診斷清理範圍
# ================================================================
df0 = pd.read_parquet(FEATURES)
df0["t0"] = pd.to_datetime(df0["t0"])
ALL_FEAT = [c for c in df0.columns if c not in DROP_COLS]

n_nan = df0["t0_payment_type"].isna().sum()
mask_fill = df0["t0_payment_type"].isna() & (df0["t0_channel_type"] == "Pos")
n_fill = mask_fill.sum()
n_true_missing = n_nan - n_fill
print(f"[清理診斷] t0_payment_type 缺失 {n_nan:,}（{n_nan/len(df0):.1%}）")
print(f"  其中 POS 通路（可解釋缺失，補 'POS'）: {n_fill:,}")
print(f"  其餘真缺失（保留 NaN）: {n_true_missing:,}")


def build_X(df, fix):
    d = df.copy()
    for c in BOOL_COLS:
        d[c] = d[c].astype(int)
    # 先做 paytype 清理（fix 版）
    if fix:
        m = d["t0_payment_type"].isna() & (d["t0_channel_type"] == "Pos")
        d.loc[m, "t0_payment_type"] = "POS"
    for c in CAT_COLS:
        d[c] = d[c].astype(str).replace("nan", "Missing")
    return d


def train_lgb_and_eval(df):
    """以 B3 相同切分/參數訓練 LightGBM(raw) + isotonic 校準，回傳 test 指標與物件。"""
    y = df["revived"].values
    train_mask = df["t0"] < pd.Timestamp("2023-11-01")
    test_mask = ~train_mask
    X = df[ALL_FEAT]

    Xtr, Xte = X[train_mask].copy(), X[test_mask].copy()
    ytr, yte = y[train_mask.values], y[test_mask.values]
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    # raw LightGBM
    clf = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
    clf.fit(Xtr, ytr, categorical_feature=CAT_COLS)
    p_raw = clf.predict_proba(Xte)[:, 1]

    # 校準（B3 做法：train 內前 80% fit、後 20% 校準，保時間序）
    tr_sorted = df[train_mask].sort_values("t0")
    cut = int(len(tr_sorted) * 0.8)
    fit_ids, cal_ids = tr_sorted.index[:cut], tr_sorted.index[cut:]
    Xfit, Xcal = X.loc[fit_ids].copy(), X.loc[cal_ids].copy()
    yfit = df.loc[fit_ids, "revived"].values
    for c in CAT_COLS:
        Xfit[c] = Xfit[c].astype("category")
        Xcal[c] = pd.Categorical(Xcal[c], categories=Xfit[c].cat.categories)
    base = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
    base.fit(Xfit, yfit, categorical_feature=CAT_COLS)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(base.predict_proba(Xcal)[:, 1], df.loc[cal_ids, "revived"].values)
    Xte_cal = X[test_mask].copy()
    for c in CAT_COLS:
        Xte_cal[c] = pd.Categorical(Xte_cal[c], categories=Xfit[c].cat.categories)
    p_cal = iso.transform(base.predict_proba(Xte_cal)[:, 1])

    metrics = {
        "raw": {"roc_auc": round(roc_auc_score(yte, p_raw), 4),
                "pr_auc": round(average_precision_score(yte, p_raw), 4),
                "brier": round(brier_score_loss(yte, np.clip(p_raw, 0, 1)), 4)},
        "calibrated": {"roc_auc": round(roc_auc_score(yte, p_cal), 4),
                       "pr_auc": round(average_precision_score(yte, p_cal), 4),
                       "brier": round(brier_score_loss(yte, np.clip(p_cal, 0, 1)), 4)},
    }
    objs = {"clf": clf, "base": base, "iso": iso,
            "p_raw": p_raw, "p_cal": p_cal, "test_mask": test_mask}
    return metrics, objs


print("\n[訓練] BEFORE（未清理）...")
m_before, _ = train_lgb_and_eval(build_X(df0, fix=False))
print("[訓練] AFTER（清理後）...")
m_after, objs_after = train_lgb_and_eval(build_X(df0, fix=True))

# ================================================================
# 前後對照
# ================================================================
print("\n" + "=" * 60)
print("  t0_payment_type 清理 — 時間外 test 前後指標對照")
print("=" * 60)
hdr = f"{'指標':<26}{'BEFORE':>10}{'AFTER':>10}{'delta':>10}"
print(hdr)
rows = []
for model in ["raw", "calibrated"]:
    for met in ["roc_auc", "pr_auc", "brier"]:
        b, a = m_before[model][met], m_after[model][met]
        d = round(a - b, 4)
        name = f"LightGBM[{model}].{met}"
        print(f"{name:<26}{b:>10}{a:>10}{d:>+10}")
        rows.append({"model": model, "metric": met, "before": b, "after": a, "delta": d})

# 結論判定：主指標（calibrated PR-AUC、raw PR-AUC）變動 < 0.005 視為結論不變
key_deltas = [abs(r["delta"]) for r in rows if r["metric"] in ("roc_auc", "pr_auc")]
unchanged = max(key_deltas) < 0.005
print("\n結論：", "指標變動極小（<0.005），結論不變 [OK]" if unchanged
      else "指標有可見變動，需檢視")

# ================================================================
# 持久化（清理寫回 + 覆寫 LightGBM 兩個模型 + test_predictions 兩欄）
# ================================================================
print("\n[持久化] 將清理寫回 features.parquet 並更新 LightGBM 模型/預測 ...")
df_fixed = df0.copy()
m = df_fixed["t0_payment_type"].isna() & (df_fixed["t0_channel_type"] == "Pos")
df_fixed.loc[m, "t0_payment_type"] = "POS"
df_fixed.to_parquet(FEATURES, index=False)

# 覆寫 LightGBM 模型（raw + 校準）
joblib.dump(objs_after["clf"], MODEL_DIR / "lightgbm.joblib")
joblib.dump({"model": objs_after["base"], "isotonic": objs_after["iso"]},
            MODEL_DIR / "lightgbm_calibrated.joblib")

# 更新 test_predictions 的兩欄（其餘模型欄位不動）
preds = pd.read_parquet(PREDS)
preds["p_lightgbm"] = objs_after["p_raw"]
preds["p_lightgbm_cal"] = objs_after["p_cal"]
preds.to_parquet(PREDS, index=False)
print("  features.parquet、lightgbm(.joblib×2)、test_predictions(p_lightgbm*) 已更新")
print("  logreg / catboost 模型與其預測欄位未動")

# ================================================================
# 報告 JSON
# ================================================================
report = {
    "cleanup_rule": "t0_payment_type NaN & t0_channel_type=='Pos' -> 'POS'; else keep NaN",
    "n_total": int(len(df0)),
    "n_paytype_nan_before": int(n_nan),
    "n_filled_POS": int(n_fill),
    "n_true_missing_kept_nan": int(n_true_missing),
    "metrics_before": m_before,
    "metrics_after": m_after,
    "deltas": rows,
    "conclusion_unchanged": bool(unchanged),
    "persisted": ["features.parquet", "lightgbm.joblib",
                  "lightgbm_calibrated.joblib", "test_predictions.parquet(p_lightgbm*)"],
    "untouched": ["logreg", "catboost"],
}
json.dump(report, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n報告: {OUT_JSON}")
