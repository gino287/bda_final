# B 組 報告資產索引（依 Sprint 分層）

> 🎯 **主敘事底稿：`output/B/MASTER_BRIEF.md`**（看一份掌握全研究 + 預期問答）。
> 🥇 **數字單一真實來源：`output/B/CANONICAL_NUMBERS.md`**（所有引用以此為準）。
> 📄 單頁總報告：`output/B/B_FINAL_REPORT.md`（B0~B5 + 四項強化任務一次看完）。
> 📖 商業故事骨架：`output/B/BUSINESS_STORY.md` + `output/B/value_funnel.png`（六段成線，報告主軸）。
> 🧭 資料夾結構：`output/B/README.md`（每資料夾另有自己的 README）。

## Block 3+2+整合（商業故事線）
- `B4/economics/ECONOMIC_NARRATIVE.md` — 三層經濟敘事（含 Pre-task 單位修正）
- `B4/economics/pretask_consistency.json` — Pre-task 一致性修正數據
- `B4/insights/INSIGHTS.md` + `insight1~5_*.png` — Block3 五條洞察（含破除迷思）
- `B5/segmentation/tier_table.csv` `strategy_map.png` `brand_deployment.md` `PLAYBOOK.md` — Block2 分級/地圖/部署
- `BUSINESS_STORY.md` + `value_funnel.png` — 最終整合單一商業故事 + 帶價值漏斗

## output/B/B0/ — 母體驗數
- `b0_summary.json` — n_usable、基本率、時間範圍、子群分佈

## output/B/B1/ — 事件表與標籤
- `event_table.parquet` — 一人一列，含 revived 標籤

## output/B/B2/ — 特徵工程
- `features.parquet` — 144,919 × 54 欄

## output/B/B3/ — 建模
- `models/` — logreg / lightgbm / catboost / lightgbm_calibrated
- `cv_results.json` — 模型比較（含三基準）
- `test_predictions.parquet` — 時間外 test 各模型機率
- `lgb_feature_importance.csv` — LightGBM gain 重要度

## output/B/B4/ — 評估與商業價值
- `figures/roc_pr.png` — ROC 與 PR 曲線
- `figures/lift_gain.png` — 累積 gain / 營收涵蓋 + decile lift
- `figures/calibration.png` — 校準可靠度曲線
- `figures/profit_curve.png` — Profit curve（全母體，原版）
- `figures/profit_curve_reachable.png` — Profit curve（可觸及者）
- `figures/profit_uplift_vs_u.png` — 修正版：期望利潤 vs uplift u（任務1）
- `figures/profit_cannibalization.png` — 修正版：margin cannibalization（任務1）
- `figures/t0_amount_revival.png` — t0_amount 分桶復活率（控制 is_imputed，任務3）
- `figures/shap_beeswarm.png` / `shap_dependence.png` — SHAP 解釋
- `business_findings.md` — 商業洞察（含附錄 A：任務 1/3/4）
- `b4_eval_summary.json` / `b_profit_revised.json` — 評估與修正 profit 數據
- `t0_amount_bins.csv` — t0_amount 分桶數據（任務3）
- `t0_paytype_cleanup.json` — t0_payment_type 清理前後對照（任務4）
- `logreg_coefficients.csv` — 邏輯迴歸係數
- `economics/ECONOMIC_NARRATIVE.md` — 三層經濟敘事（**三層全部完成**）
- `economics/layer1_margin_saving.json` — 第一層：高分層停發券省 margin + 損益破口
- `economics/layer2_crossover.json` — 第二層：uplift 破口 + 產業實證區間
- `economics/layer3_robustness.json` — 第三層：96 組跨假設掃描 + 因果限制
- `figures/layer1_saving_by_tier.png` — 第一層：分數層復活率 + 停發省下 vs 風險
- `figures/layer2_profit_vs_u.png` — 第二層：四線 profit vs u + 產業 uplift 帶
- `figures/layer2_sensitivity_heatmap.png` — 第二層：毛利率 × 兌換率 敏感度
- `figures/layer3_win_heatmap.png` — 第三層：u=0.03 模型對全發券優勢熱圖

## output/B/B5/ — 穩健性、子群、報告
- `figures/funnel.png` — 沉睡→回流→復活/路過 漏斗
- `b5_subgroup_metrics.csv` — is_imputed / ShopId 子群指標
- `b5_robustness.json` — 替代復活定義穩健性檢驗
- `report_assets/B5_report.md` — 穩健性/子群/漏斗/限制/賣點
- `report_assets/brand_heterogeneity.csv` / `.png` / `.json` — 品牌異質性表（依 base rate 排序，驗證低復活率品牌 targeting 價值最大）
