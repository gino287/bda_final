# B3/robustness_features — 任務 B-R1 特徵 ablation

由 `src/B/b3c_feature_ablation.py`（任務 B-R1）產生。回應「ShopId、MemberCardLevel 不該餵模型」的疑慮。
**未動 frozen 模型**；四變體 raw LightGBM、同一份時間外 test。

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `feature_ablation.json` | 四變體（全特徵 / 移除兩者 / 各別移除）ROC-AUC·PR-AUC·Brier | ✅ 現行有效 |
| `feature_ablation.md` | 解讀 + 一句話回應質疑 | ✅ 現行有效 |

結論：移除兩者 PR-AUC 0.569→0.542（仍 > RFM-logreg 0.503）；ShopId 單獨移除略變好；MemberCardLevel leakage 影響有界（ΔPR −0.025）。詳見 `../../CANONICAL_NUMBERS.md` 第 3 節。
