# B-R2 數字同步報告（sync_report）

> 目的：把所有對外 .md 的數字對齊 `CANONICAL_NUMBERS.md`（單一真實來源）。
> 本次**只統一數字、不改分析結論**。日期：2026-06-04。

## 變更清單

| 檔案 | 位置 | 舊值（已作廢） | 新值（對齊 canonical） |
|------|------|------|------|
| `B_FINAL_REPORT.md` | B3 模型表 | baseline「RFM 規則」0.562 / 0.392 | **RFM-only 邏輯迴歸 0.645 / 0.503** |
| `B_FINAL_REPORT.md` | B3 驗收句 | 「vs 基準最高 0.39，+45%」 | 「vs RFM-logreg 0.503，**+15%**」 |
| `B_FINAL_REPORT.md` | B4 原版 Profit | 僅標「已由任務1修正」 | 補：已被 uplift 修正版取代為主敘事，指向 ECONOMIC_NARRATIVE / CANONICAL |
| `B4/business_findings.md` | 第1節對照基準 | 「RFM 規則 PR-AUC≈0.39」 | 「**RFM-only 邏輯迴歸 0.503**」+ 贏 +15% |
| `B4/business_findings.md` | 第5節洞察 #2 | 「較高 t0 金額者更可能復活」（方向錯） | 「**t0 金額越高、復活率越低**（−0.90）」 |
| `B4/economics/ECONOMIC_NARRATIVE.md` | 第二層 grid 表 | u=0.03 列、全發券「−2,228,000」 | u=0.15（產業比例下緣）列；u 標明為**比例** |
| `B4/economics/ECONOMIC_NARRATIVE.md` | 第二層 詳述 | 「產業 uplift 帶 1.2%~3.0%」「全發券利潤為負」 | 「產業**比例**帶 15%~25%」「落損益兩平**微虧側**、模型多賺 +0.7~1.3M」 |
| `B4/economics/ECONOMIC_NARRATIVE.md` | 第二層 兌換率註 | 「兌換率 0.40→8~15%」（含糊） | 明確：決策經濟用即時觸發 0.40；L1 硬省用保守 8~15% |
| `B4/economics/ECONOMIC_NARRATIVE.md` | 第三層 指標表 | 模型勝 RFM 79/96、treat-all 虧 46/96 | **96/96、22/96**（單位修正後） |
| `ENHANCEMENTS.md` | 影響彙整表 | B_FINAL_REPORT「⚠️待同步 0.39」 | 「✅已同步（B-R2）」 |

## 已於更早任務修正、本次確認無殘留
- 折扣洞察：全部使用「**破除迷思**」版本（t0 折扣深度非流失訊號，Spearman +0.66），無殘留舊假說「折扣驅動較不黏著」當作結論。`INSIGHTS.md` 第 8 行為「假說陳述後推翻」的正確框架。
- t0_amount 方向：`business_findings.md` / `ENHANCEMENTS.md` 的 ⚠️ 更正段為「引用舊錯誤以更正」，屬合法保留。
- Pre-task 修正記錄（ECONOMIC_NARRATIVE 頂部 46/96→22/96 等）為「舊→新對照表」，刻意保留以示修正軌跡。

## 未改動（刻意保留）
- `B4/economics/ECONOMIC_NARRATIVE_updated.md`：使用者提供之**輸入規格 / 產業錨點來源（附引用）**，非報告；保留供查證，狀態見 `economics/README.md`。
- 各 JSON / CSV：由腳本產生，為數字之根；canonical 由其抽取，未手改。
- 原版 profit 圖（`profit_curve*.png`）：保留為早期「全額毛利」框架對照，正文已標註被取代。

## 結論
所有對外 .md 數字已與 `CANONICAL_NUMBERS.md` 一致；分析結論未改動。
