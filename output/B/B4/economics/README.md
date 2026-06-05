# B4/economics — 三層經濟敘事（Block 1 + Pre-task 修正）

腳本：`b4d_layer1_*`（L1）、`b4e_layer2_*`（L2）、`b4f_layer3_*`（L3）、`b4g_pretask_consistency.py`（單位修正）。

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `ECONOMIC_NARRATIVE.md` | **三層完整敘事（含 Pre-task 單位修正）** | ✅ 現行主敘事（數字已同步 canonical） |
| `pretask_consistency.json` | Pre-task：u(比例)單位修正、兌換率區間、L3 重算 | ✅ 現行 |
| `layer1_margin_saving.json` | L1 高分層停發券省 margin + 損益破口 | ✅ 現行（注意：兌換率敏感度見 pretask json） |
| `layer2_crossover.json` | L2 uplift 破口 + 產業帶（**注意：u 帶以 pretask 修正版 15~25% 為準**） | ✅ 現行（u 帶數字以 canonical/pretask 為準） |
| `layer3_robustness.json` | L3 96 組掃描（**注意：treat-all 虧損組數以 pretask 修正版 22/96 為準**） | ✅ 現行（虧損組數以 canonical/pretask 為準） |
| `ECONOMIC_NARRATIVE_updated.md` | **使用者輸入規格 / 產業錨點來源（附引用）**，非報告 | 📎 輸入參考，保留供查證；敘事以 ECONOMIC_NARRATIVE.md 為準 |

> 單位修正重點：模型 u 是**比例 ratio**（非絕對 pp）。產業 winback uplift 換算後 = 15~25%。
> `layer2_crossover.json` / `layer3_robustness.json` 是 Pre-task **之前**的初版掃描；其 u 帶（1.2~3%）與 L3 虧損組數（46/96）已被 `pretask_consistency.json` 修正為 15~25% 與 22/96。**引用一律以 `../../CANONICAL_NUMBERS.md` 第 4 節為準。**
