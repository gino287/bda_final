# B4 — 評估與商業價值

Sprint B4 + 多項強化任務的產出。

## 評估核心（`src/B/b4_evaluate.py`）
| 檔案 | 說明 | 狀態 |
|------|------|------|
| `business_findings.md` | 商業洞察（含附錄 A：任務 1/3/4） | ✅ 現行（數字已同步 canonical） |
| `b4_eval_summary.json` | ROC/PR/Brier、top-decile capture、**原版 profit** | ⚠️ 原版 profit 為早期「全額毛利」框架，已被 economics/ 的 uplift 修正版取代為主敘事 |
| `logreg_coefficients.csv` | 邏輯迴歸係數表 | ✅ 現行 |

## 強化任務產出
| 檔案 | 任務 | 說明 | 狀態 |
|------|------|------|------|
| `b_profit_revised.json` | 任務1 | uplift + cannibalization 修正 profit | ✅ 現行 |
| `t0_amount_bins.csv` | 任務3 | t0 金額分桶復活率 | ✅ 現行 |
| `t0_paytype_cleanup.json` | 任務4 | t0_payment_type 清理前後對照 | ✅ 現行 |
| `economics/` | Block1 | 三層經濟敘事（見其 README） | ✅ 現行（主經濟敘事） |
| `insights/` | Block3 | 五條洞察（見其 README） | ✅ 現行 |
| `figures/` | 各 | 所有圖（見其 README） | ✅ 現行 |

關鍵數字見 `../CANONICAL_NUMBERS.md` 第 2、4 節。
