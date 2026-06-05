# B4/figures — 圖表

| 檔案 | 由誰產生 | 說明 | 狀態 |
|------|---------|------|------|
| `roc_pr.png` | b4_evaluate | ROC 與 PR 曲線 | ✅ 現行 |
| `lift_gain.png` | b4_evaluate | 累積 gain / 營收涵蓋 + decile lift | ✅ 現行 |
| `calibration.png` | b4_evaluate | 校準可靠度曲線 | ✅ 現行 |
| `shap_beeswarm.png` | b4_evaluate | SHAP 全域重要度 | ✅ 現行 |
| `shap_dependence.png` | b4_evaluate | SHAP dependence | ✅ 現行 |
| `t0_amount_revival.png` | 任務3 | t0 金額分桶復活率（控制 is_imputed） | ✅ 現行 |
| `profit_curve.png` | b4_evaluate | 原版 profit（全額毛利框架） | ⚠️ 早期對照，主敘事用 economics/ uplift 版 |
| `profit_curve_reachable.png` | b4_evaluate | 原版 profit（可觸及者） | ⚠️ 早期對照 |
| `profit_uplift_vs_u.png` | 任務1 | uplift profit vs u | ✅ 現行（任務1版） |
| `profit_cannibalization.png` | 任務1 | margin cannibalization | ✅ 現行（任務1版） |
| `layer1_saving_by_tier.png` | Block1-L1 | 分數層復活率 + 停發省下 vs 風險 | ✅ 現行 |
| `layer2_profit_vs_u.png` | Block1-L2（pretask 重畫） | 四線 + 產業**比例**帶 15~25% + 破口 | ✅ 現行 |
| `layer2_sensitivity_heatmap.png` | Block1-L2 | 毛利率 × 兌換率 敏感度 | ✅ 現行 |
| `layer3_win_heatmap.png` | Block1-L3（pretask 重畫） | u=0.20 模型對全發券優勢熱圖 | ✅ 現行 |
| `準確率比較圖.png` | **外部/使用者提供** | 非本 pipeline 產生、用途未知 | ❓ 保留未動（如不需要可自行刪除） |

> 經濟主敘事的權威圖為 `layer1/2/3` 系列；原版 `profit_curve*.png` 僅早期對照。
