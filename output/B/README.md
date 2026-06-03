# output/B/ — B 組產出資料夾結構

> 📄 **想一次看完整結論 → `B_FINAL_REPORT.md`**（單頁總報告，串接 B0~B5 + 四項強化任務）。

依 Sprint 分層，方便按階段查找。每個檔案歸到「產生它的 Sprint」。

```
output/B/
├── B0/   母體驗數
│   └── b0_summary.json
├── B1/   事件表與標籤
│   └── event_table.parquet
├── B2/   特徵工程
│   └── features.parquet               (144,919 × 54)
├── B3/   建模
│   ├── models/                        logreg, lightgbm, catboost, lightgbm_calibrated
│   ├── cv_results.json
│   ├── test_predictions.parquet
│   └── lgb_feature_importance.csv
├── B4/   評估與商業價值
│   ├── figures/                       roc_pr, lift_gain, calibration,
│   │                                  profit_curve(_reachable),
│   │                                  profit_uplift_vs_u, profit_cannibalization,
│   │                                  t0_amount_revival,
│   │                                  shap_beeswarm, shap_dependence
│   ├── business_findings.md           (含附錄 A：任務 1/3/4)
│   ├── b4_eval_summary.json
│   ├── b_profit_revised.json          修正版 profit（uplift + cannibalization）任務1
│   ├── t0_amount_bins.csv             t0_amount 分桶數據（任務3）
│   ├── t0_paytype_cleanup.json        t0_payment_type 清理前後對照（任務4）
│   └── logreg_coefficients.csv
└── B5/   穩健性、子群、報告
    ├── figures/funnel.png
    ├── b5_subgroup_metrics.csv
    ├── b5_robustness.json
    └── report_assets/                 B5_report.md (含附錄 B：任務2), INDEX.md,
                                       brand_heterogeneity.csv/.png/.json (任務2)
```

## 對應腳本（src/B/）
| Sprint | 腳本 | 產出資料夾 |
|--------|------|-----------|
| B0 | `b0_sanity.py` | `output/B/B0/` |
| B1 | `b1_label.py` | `output/B/B1/` |
| B2 | `b2_features.py` | `output/B/B2/` |
| B3 | `b3_train.py` | `output/B/B3/` |
| B4 | `b4_evaluate.py` + `b4b_profit_revised.py` | `output/B/B4/` |
| B5 | `b5_robustness_report.py` | `output/B/B5/` |

## 後續強化任務（第一版完成後追加）
| 任務 | 腳本 | 產出 | 報告位置 |
|------|------|------|---------|
| 1 修正版 profit（uplift+cannibalization） | `b4b_profit_revised.py` | B4/b_profit_revised.json、2 張 profit 圖 | business_findings.md 附錄 A1 |
| 2 品牌異質性表 | `b5b_brand_heterogeneity.py` | B5/report_assets/brand_heterogeneity.* | B5_report.md 附錄 B |
| 3 t0_amount 洞察佐證 | `b4c_t0amount_insight.py` | B4/figures/t0_amount_revival.png、B4/t0_amount_bins.csv | business_findings.md 附錄 A2（更正洞察#2方向） |
| 4 t0_payment_type 清理 | `b3b_paytype_check.py` | B4/t0_paytype_cleanup.json（已寫回 features + lightgbm 模型） | business_findings.md 附錄 A3 |

## 資料流（重跑順序）
B0 → B1 → B2 → B3 → B4（含 b4b/b4c）→ B5（含 b5b）。
後段 sprint 讀取前段產出（如 B4 讀 B2/features + B3/models、test_predictions）。
任務 4 的清理已併入 `b2_features.py`，未來重跑 B2 即自帶；任務 3/4 腳本可於 B2/B3 完成後獨立執行。
