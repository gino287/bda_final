# output/B/ — B 組產出總導覽

> 🎯 **看一份就掌握全研究 → `MASTER_BRIEF.md`**（四段成線 + 預期問答，現場底稿）。
> 🥇 **數字一律以 `CANONICAL_NUMBERS.md` 為準**（單一真實來源，2026-06-04 同步）。
> 📖 報告骨架 → `BUSINESS_STORY.md` + `value_funnel.png`（六段商業故事）。
> 📄 技術總報告 → `B_FINAL_REPORT.md`（B0~B5 + 強化任務）。
> 🧭 全資產索引 → `B5/report_assets/INDEX.md`。

## 頂層文件（先看這些）

| 檔案 | 用途 |
|------|------|
| **`MASTER_BRIEF.md`** | 🎯 看一份就掌握全研究：四段成線 + 預期問答（現場底稿） |
| **`CANONICAL_NUMBERS.md`** | ★ 所有對外數字的單一真實來源，每個標來源檔 |
| `BUSINESS_STORY.md` | 六段商業故事（報告主軸）+ `value_funnel.png` 總圖 |
| `B_FINAL_REPORT.md` | 技術總報告（B0~B5 + 四強化任務） |
| `ENHANCEMENTS.md` | 強化任務（1~5）變更紀錄 |
| `sync_report.md` | B-R2 數字同步變更清單 |
| `README.md` | 本檔（總導覽） |

## 資料夾結構（每個資料夾都有自己的 README.md）

```
output/B/
├── CANONICAL_NUMBERS.md  ★ 數字單一真實來源
├── BUSINESS_STORY.md     報告骨架（+ value_funnel.png）
├── B_FINAL_REPORT.md     技術總報告
├── ENHANCEMENTS.md / sync_report.md
├── B0/  母體驗數          → b0_summary.json
├── B1/  事件表與標籤       → event_table.parquet
├── B2/  特徵工程          → features.parquet (144,919×54)
├── B3/  建模              → models/, cv_results.json, test_predictions.parquet
│   └── robustness_features/   任務 B-R1 特徵 ablation
├── B4/  評估與商業價值
│   ├── business_findings.md, b4_eval_summary.json, b_profit_revised.json ...
│   ├── economics/         三層經濟敘事（ECONOMIC_NARRATIVE.md = 主敘事）
│   ├── insights/          Block3 五條洞察（INSIGHTS.md）
│   └── figures/           所有圖
├── B5/  穩健性/子群/報告/分級
│   ├── report_assets/     B5_report.md, brand_heterogeneity.*, INDEX.md
│   └── segmentation/      Block2 分級/策略地圖/部署/PLAYBOOK
└── _archive/              已取代/廢棄檔（目前無，見其 README）
```

> 每個資料夾的 `README.md` 逐檔說明「這是什麼、哪個 sprint/task 產的、現行有效或已被取代」。

## 對應腳本（src/B/）

| 階段 | 腳本 |
|------|------|
| 主流程 B0~B5 | `b0_sanity` `b1_label` `b2_features` `b3_train` `b4_evaluate` `b5_robustness_report` |
| 強化任務 1~5 | `b4b_profit_revised`(1) `b5b_brand_heterogeneity`(2) `b4c_t0amount_insight`(3) `b3b_paytype_check`(4) `b3_train`(5 baseline) |
| Block1 三層經濟 | `b4d_layer1_margin_saving` `b4e_layer2_crossover` `b4f_layer3_robustness` `b4g_pretask_consistency` |
| Block3 / Block2 / 整合 | `b4h_insights` / `b5c_segmentation` / `b_business_story` |
| 任務 B-R1 ablation | `b3c_feature_ablation` |

## 資料流（重跑順序）
B0 → B1 → B2 → B3 →（B-R1 ablation）→ B4（評估 + 強化任務 + Block1 三層 + 洞察）→ B5（穩健 + Block2 分級）→ 整合（BUSINESS_STORY）。
後段讀前段產出。所有 leakage 鐵律於 B2 把關；frozen 模型在 B3/models。

## 環境
所有腳本用 `./venv/Scripts/python` 執行（全域 python 無 duckdb）。
