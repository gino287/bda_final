# B3 — 建模

由 `src/B/b3_train.py`（Sprint B3）產生。baseline 已於任務 5 改為 RFM-only 邏輯迴歸。

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `models/logreg.joblib` | 邏輯迴歸（全特徵，標準化+補值+one-hot） | ✅ frozen |
| `models/lightgbm.joblib` | LightGBM raw（任務4後重訓，與舊版等價） | ✅ frozen 主力 |
| `models/lightgbm_calibrated.joblib` | LightGBM + isotonic 校準（**決策機率來源**） | ✅ frozen 主決策 |
| `models/catboost.cbm` | CatBoost（PR-AUC 最佳 0.578） | ✅ frozen |
| `cv_results.json` | 八模型/基準比較（時間外 test） | ✅ 現行有效 |
| `test_predictions.parquet` | 時間外 test 各模型機率 + 標籤 | ✅ 現行有效（下游分析輸入） |
| `lgb_feature_importance.csv` | LightGBM gain 重要度 | ✅ 現行有效 |
| `robustness_features/` | 任務 B-R1：ShopId/MemberCardLevel ablation（見其 README） | ✅ 現行有效 |

> **frozen 模型勿覆蓋**。任務 B-R1 的 ablation 變體另存於 `robustness_features/`，未動 frozen 模型。

關鍵數字見 `../CANONICAL_NUMBERS.md` 第 2 節。
