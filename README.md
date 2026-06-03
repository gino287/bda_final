# 沉睡客回流分析 — 資料前處理專案

## 1. 專題概覽

**研究題目**：沉睡客回流後的消費生命週期能否被重新啟動？

**資料來源**：91APP 美妝保健產業資料集，涵蓋會員資料、訂單記錄、Web/App 行為 session 紀錄

**負責範圍**：資料前處理（Sprint 0 ~ Sprint 3），包含沉睡客識別、回流偵測、資料篩選與品質驗證

---

## 2. 前處理成果摘要

### 會員漏斗

| 層級 | 人數 | 比例 |
|------|-----:|------|
| 全體會員 | **6,904,081** 人 | — |
| 有購買記錄 | **2,217,958** 人 | 32.1% of 全體 |
| 沉睡客 | **1,112,087** 人 | 50.1% of 有購買記錄 |
| 有回流沉睡客 | **178,347** 人 | 16.0% of 沉睡客 |

### 資料縮減成果

| 階段 | 大小 | 相較原始 |
|------|-----:|--------:|
| 原始資料（CSV） | **47,305 MB** | — |
| 篩選後 CSV | **7,087 MB** | 節省 **85.0%** |
| 最終 Parquet | **3,169 MB** | 節省 **93.3%** |

---

## 3. Notebook 說明

| Notebook | 用途 | 主要輸出 | 狀態 |
|----------|------|----------|------|
| `00_sprint0_validation.ipynb` | 確認所有資料表能讀取、格式正確、表之間能 JOIN | 無 | ⚠️ 僅供參考，不需重跑 |
| `01_sprint1_member_cycle.ipynb` | 計算個人購物週期、識別沉睡客 | `output/sprint1/` | ⚠️ 不需重跑（已輸出 parquet） |
| `02_sprint2_return_detection.ipynb` | 從 session 行為資料找出回流沉睡客、計算 t0 | `output/sprint2/` | ⚠️ 不需重跑（已輸出 parquet，原始 session CSV 已刪除） |
| `03_sprint3_validation.ipynb` | 正式篩選前的驗證（樣本量、缺值率、漏斗數字） | 無 | ⚠️ 僅供參考，不需重跑 |
| `04_sprint3_filtering.ipynb` | 將所有原表篩選為只保留沉睡客資料，輸出 parquet | `output/sprint3/` | ⚠️ 不需重跑（已輸出 parquet，原始 CSV 已刪除） |

---

## 4. 輸出檔案說明

### Sprint 1 輸出（`output/sprint1/`）

| 檔案 | 說明 | 筆數 |
|------|------|-----:|
| `dormant_members.parquet` | 沉睡客名單，含購物週期、最後購買日、回流門檻日 | **1,112,087** |
| `member_cycle.parquet` | 全體有購買記錄會員的購物週期表（含 is_imputed 旗標） | **1,942,165** |

### Sprint 2 輸出（`output/sprint2/`）

| 檔案 | 說明 | 筆數 |
|------|------|-----:|
| `dormant_activity_merged.parquet` | 有回流的沉睡客，含 t0、total_events、active_days | **178,347** |
| `session01_dormant_activity.parquet` | session01 回流者彙總（合併前備用） | **141,866** |
| `session02_dormant_activity.parquet` | session02 回流者彙總（合併前備用） | **36,481** |

### Sprint 3 輸出（`output/sprint3/`）

| 檔案 | 說明 | 大小 |
|------|------|-----:|
| `member.parquet` | 沉睡客會員資料（已去除 601 筆重複，保留最早註冊） | 65 MB |
| `order_tg.parquet` | 沉睡客主單資料（訂單狀態歷程） | 267 MB |
| `order_ts.parquet` | 沉睡客子單資料（訂單明細，品項層級） | 1,207 MB |
| `session01_202309.parquet` ~ `session01_202402.parquet` | 沉睡客 Web/App 行為資料（shop #1~3） | 各 **216~292 MB** |
| `session02_202309.parquet` ~ `session02_202402.parquet` | 沉睡客 Web/App 行為資料（shop #4） | 各 **9~13 MB** |

---

## 5. 後續使用方式

### 主要工作流程

1. 以 `dormant_activity_merged.parquet` 為主表（**178,347** 人）
2. 貼上「復活 vs 路過」標籤（從 `order_tg.parquet` 中 t0 後的訂單判斷）
3. 從 `session01` / `session02` parquet 擷取行為特徵
4. 從 `member.parquet` 擷取會員人口屬性（性別、註冊來源、會員等級等）
5. 從 `order_tg.parquet` 擷取歷史購買特徵

### 讀取方式

小型資料表（member、session02）可直接用 pandas：

```python
import pandas as pd
df = pd.read_parquet('output/sprint3/member.parquet')
```

大型資料表（order_ts、session01）建議使用 DuckDB，避免 OOM：

```python
import duckdb
df = duckdb.query("SELECT * FROM 'output/sprint3/order_ts.parquet'").df()
```

---

## 6. 核心定義

| 參數 | 值 | 說明 |
|------|----|------|
| 基準日 | **2023-09-01** | 沉睡判斷的起始觀察點 |
| 沉睡倍率 | **1.5x** | 超過 1.5 個購物週期未購買即判定沉睡 |
| 週期下限（Q1） | **29 天** | IQR 法去除極端高頻購買者的下界 |
| 填補中位數 | **88.5 天** | 僅購買過一次的會員以此填補週期（is_imputed = True） |
| 復活觀察期 | **90 天** | 固定，從 t0 起算 |
| 回流定義 | session 有任何事件 | 在 session01 或 session02 中出現任何行為紀錄即視為回流 |

---

## 7. 資料品質備註

- **Member 重複**：原始資料有 **601 筆**重複 ShopMemberId（線上線下帳號合併不完整），已保留 RegisterDateTime 最早的那筆，去重後 1,112,081 人。
- **Session01 匿名用戶**：約 **32.36%** 的事件 ShopMemberId 為 NULL（未登入訪客），已於篩選時移除。
- **Session02 匿名用戶**：約 **20.00%** 的事件 ShopMemberId 為 NULL，已於篩選時移除。
- **Session01 / Session02 品牌不重疊**：session01 涵蓋 shop #1~3，session02 涵蓋 shop #4，ShopMemberId 為 per-ShopId 雜湊，兩者無法跨品牌對應，分析時需視為獨立品牌群。
- **t0 最早值 2023-08-31**：為時區轉換造成（HitTime 為 UTC 毫秒時間戳），實際對應台灣時間 2023-09-01，數值正常。

---

## 8. 後續分析：B 組「復活 vs 路過」預測建模（V2_VERSION）

本 README 涵蓋 A 組前處理（Sprint 0~3）。下游的 **B 組預測建模**（沿用本前處理產出）為獨立工作，文件與產出另置：

- **規格書**：`V2_VERSION/B組_復活預測_Sprint規格書.md`
- **產出總覽**：`output/B/README.md`（依 B0~B5 分層）
- **腳本**：`src/B/`（b0~b5 + 強化任務腳本 b3b/b4b/b4c/b5b）
- **報告**：`output/B/B4/business_findings.md`（商業洞察 + 附錄 A）、`output/B/B5/report_assets/B5_report.md`（穩健性/子群/漏斗/限制 + 附錄 B）

> 注意：B 組與 A 組**回流定義不同**。A 組用 **session 行為**判定回流；B 組改用**純訂單**（首筆 Finish 訂單 ≥ 2023-09-01）定義 t0，母體較大（232,328 vs 178,347）且可於回購當下即時觸發決策。

### B 組主要成果摘要
- B 母體 **144,919** 人（t0 ≤ 2023-12-01），復活率 **37.6%**。
- 主模型 LightGBM/CatBoost 時間外 test ROC-AUC ≈ **0.71**、PR-AUC ≈ **0.57**，明顯優於 RFM/NAPL/多數類三基準。
- 商業價值（修正版 profit）：在實務常見 uplift 10–30% 區間，全發券會虧損，**模型選擇性鎖定才獲利**；低復活率品牌 targeting 價值最大（lift 相關 −0.70）。
