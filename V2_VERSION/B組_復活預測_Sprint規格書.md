# B 組｜沉睡客「復活 vs 路過」預測 — Sprint 執行規格書

> 定位：A→B 兩階段喚回漏斗的下游。A 預測「沉睡客會不會回流」，B 預測「回流者會復活還是路過」。
> 本規格供 Claude Code 逐步執行：建構 → 貼標 → 特徵 → 訓練 → 評估 → 報告。

---

## 0. 鎖定的核心定義（不要再改，全 sprint 共用）

| 項目 | 定義 |
|------|------|
| 沉睡客母體 | 沿用 A 組既有前處理：`dormant_members.parquet`（2023-09-01 靜態基準，1,112,087 人） |
| 回流 t0 | 該沉睡客 **OrderDateTime ≥ 2023-09-01 且 StatusDef = Finish** 的「第一筆」訂單時間 |
| B 母體 | 有 t0、且 **t0 ≤ 2023-12-01** 的沉睡客（確保留得出完整 90 天觀察窗口；資料止於 2024-02-29） |
| 標籤 `revived` | `(t0, t0+90天]` 內存在 ≥1 筆 Finish → 復活(1)；否則 → 路過(0) |
| 每人入樣 | 一人一次（靜態母體 + 第一筆 t0，天然成立） |
| 回流只認 | Finish（Fail/Cancel/Return/Overdue/Shipping/New 一律不算回流，但歷史特徵可另計退貨率） |
| 預測時點 | **t0（回購當下）** |
| Leakage 鐵律 | 特徵只能用「OrderDateTime < t0 的歷史」+「t0 那筆訂單本身」。**t0 之後的任何資料都不可進特徵。** |
| 建模單位 | 「人」（ShopMemberId 不跨 ShopId，已驗證）；五品牌 **pooled，ShopId 當特徵** |
| imputed 子群 | `is_imputed=True`（只買過一次、用 88.5 天填補週期者）全程**單獨追蹤、分群報告** |

---

## 1. 資料來源對照

| 檔案 | 層級 | 關鍵欄位 | 讀取方式 |
|------|------|----------|----------|
| `output/sprint1/dormant_members.parquet` | 人 | ShopMemberId, purchase_count, last_purchase_date, is_imputed, personal_cycle_days, dormant_threshold_date, days_since_last | pandas |
| `output/sprint3/member.parquet` | 人 | Gender, Birthday, RegisterSourceTypeDef, RegisterDateTime, IsAppInstalled, IsEnable{Email,PushNotification,ShortMessage}, MemberCardLevel, First/LastAppOpenDateTime, ShopId | pandas |
| `output/sprint3/order_tg.parquet` | 主單 | ShopMemberId, OrderDateTime, StatusDef, TotalSalesAmount, ChannelType, ChannelDetail, TradesGroupCode | DuckDB |
| `output/sprint3/order_ts.parquet` (1.2GB) | 子單 | ShopMemberId, OrderDateTime, SalePageId, Qty, UnitPrice, SubtotalPrice, 折扣三欄, PaymentType, TradesGroupCode | **務必 DuckDB，勿進 pandas** |

> 折扣三欄（子單）：`SubtotalPromotionDiscount`、`SubtotalCouponDiscount`、`SubtotalLoyaltyPointDiscount`（皆為負值或 0）。

---

## Sprint B0 — 資料健檢與母體驗數 ★ 最優先

**目標**：確認 B 母體規模、資料時間範圍、基本率，任何異常在此攔下。

**步驟**
1. 用 DuckDB 從 `order_tg` 算每人 t0 = `MIN(OrderDateTime) WHERE StatusDef='Finish' AND OrderDateTime >= '2023-09-01'`。
2. 左接 `dormant_members`（只保留沉睡客），過濾 `t0 <= '2023-12-01'` → 得 B 母體，輸出 `n_usable`。
3. 確認 `order_tg` 歷史回溯起點 `MIN(OrderDateTime)`（預期 ~2022-01，決定特徵回看期長度）。
4. 貼標後算 **基本率**：復活佔比（預期路過為多數）。
5. 分 `is_imputed`、分 `ShopId` 各看一次母體分佈。

**輸出**：`output/B/b0_summary.json`（n_usable、基本率、時間範圍、各子群人數）

**驗收**
- `n_usable` ≥ 上萬 → 照計畫走。
- 若 `n_usable` 很小（< 數千）或復活率極端（<2% 或 >50%）→ 停下回報，調整 X 或 t0 上限。
- `order_tg` 確認有 2022 年資料。

---

## Sprint B1 — 事件表與標籤 ★ MUST

**目標**：產出 B 主表（一人一列）。

**步驟**
1. 建事件表：`ShopMemberId, ShopId, t0, is_imputed`。
2. 貼標：DuckDB 查 `(t0, t0+INTERVAL 90 DAY]` 內是否有 Finish → `revived ∈ {0,1}`。
3. 同時記錄（給後續 profit 用，**不可當特徵**）：復活者在該 90 天窗口內的 Finish 總營收 `future_90d_revenue`。
4. 加一個 leakage 防呆斷言：事件表不得含任何 ≥ t0 的衍生欄位（除 t0 本身與 future_* 標籤欄）。

**輸出**：`output/B/event_table.parquet`（含 revived、future_90d_revenue）

**驗收**：每人唯一、revived 無缺值、t0 全部 ≤ 2023-12-01。

---

## Sprint B2 — 特徵工程 ★ MUST（核心） + ◇（加分）

**鐵律**：歷史特徵一律 `WHERE OrderDateTime < t0`；t0 訂單特徵取 `OrderDateTime == t0` 那筆（同日多筆則彙總）。每組特徵產出後跑一次 leakage 斷言。

### 2A. 歷史 RFM 與購買行為（order_tg / order_ts，< t0）★
- Recency：t0 − 上一筆 Finish 日期（沉睡深度）。
- Frequency：歷史 Finish 訂單數；近 365 天 Finish 數。
- Monetary：歷史總額、AOV（均）、中位數、標準差。
- Tenure：t0 − RegisterDateTime（年資）。
- 週期：`personal_cycle_days`、`is_imputed`、購買間隔變異係數（CV，規律性）。◇
- 趨勢：後半段消費額 / 前半段消費額（加速/減速）。◇
- 沉睡前空窗：t0 − last_purchase_date（離開多久才回來）。

### 2B. 品類與折扣（order_ts，< t0）★ 核心折扣依賴 / ◇ 品類
- 折扣依賴度：歷史 (促銷+折價券+會員折扣) / 銷售金額 之平均佔比；有折扣訂單比例。★
- 品類廣度：distinct SalePageId 數；品類熵（多樣性）。◇
- 退貨率：Return 訂單數 / 成立訂單數。◇

### 2C. 通路（order_tg，< t0）◇
- 線上/線下佔比（ChannelType）、App/Web/POS 佔比（ChannelDetail）、使用過的通路數。

### 2D. t0 回購訂單本身 ★ 通常最強，且合法
- t0 訂單金額（TotalSalesAmount）。
- t0 籃子：品項數（TsCount / distinct SalePageId）、總 Qty。
- t0 折扣深度（該筆折扣佔比）。★
- t0 通路（ChannelType, ChannelDetail）。
- t0 付款方式（PaymentType：一次/分期/先買後付 等）。◇
- t0 品類是否為「歷史買過的舊品類」或「全新品類嘗試」。◇

### 2E. 會員屬性（member.parquet）★
- `MemberCardLevel`（0 視為缺失 → NaN + 缺失旗標）。
- 行銷可觸及性：`IsEnablePushNotification / Email / ShortMessage`，及「啟用通路數」。★ 商業關鍵
- `IsAppInstalled`；App 互動：t0 − LastAppOpenDateTime。◇
- `RegisterSourceTypeDef`、`ShopId`、`Gender`（null→"Unknown"）、年齡（Birthday 過濾 1900 預設值後分桶）。

### 缺失值處理
- `1900-01-01` 預設值（Birthday/RegisterDateTime）→ 一律轉 NaN，年齡/年資算前先濾。
- 樹模型（LightGBM/CatBoost）原生吃 NaN；邏輯迴歸需顯式補值 + 缺失旗標。

**輸出**：`output/B/features.parquet`（事件表左接全部特徵）

**驗收**：leakage 斷言全過；特徵缺失率報表；類別欄位型別正確。

---

## Sprint B3 — 建模 ★ MUST

**目標**：訓練並比較模型，且要贏過簡單基準。

**步驟**
1. **切分（時間外驗證）**：依 t0 排序，早期 t0 當 train（如 9–10 月）、晚期當 test（11–12 月初）→ 模擬「用過去預測未來」。另存一份 stratified 隨機 80/20 當次要對照。
2. **不平衡處理**：先看基本率；用 `class_weight` / `scale_pos_weight`，**不做 resample**（保住機率校準）。評估靠排序型指標，不看 accuracy。
3. **模型清單**
   - 邏輯迴歸（標準化 + 可解釋 baseline）。
   - **LightGBM 或 XGBoost**（主力效能）。
   - **CatBoost**（原生吃類別欄位，本資料類別多，建議納入）。◇
4. **調參**：Optuna 時間盒（small trials），以 validation 的 PR-AUC 為目標，搭配時間感知 CV。◇
5. **校準**：若 reliability 偏移，用 isotonic / Platt 重校（機率要拿來決定花錢，校準重要）。
6. **必備基準對照**（要打贏才算貢獻）
   - 多數類。
   - RFM 規則（recency + monetary 門檻）。
   - NAPL 領域規則（3× 週期/卡等門檻）。

**輸出**：`output/B/models/`（模型檔 + 預測機率）、`output/B/cv_results.json`

**驗收**：主模型在時間外 test 的 AUC/PR-AUC 明確優於三個基準。

---

## Sprint B4 — 評估與商業價值 ★ MUST（拿分核心）

**目標**：把模型翻譯成商業決策，不要只丟 AUC。

**步驟**
1. **辨識力**：ROC-AUC、PR-AUC（不平衡看後者）。
2. **排序/商業**：gain & lift 圖、top-decile capture；**累積營收涵蓋率**（用 future_90d_revenue 加權）——「鎖定前 20% 能涵蓋 X% 未來復活營收」。
3. **校準**：reliability curve + Brier。
4. **Profit curve（最重要一張圖）**
   - 設定單位經濟（先確認假設，見下節）：每位真復活的預期增量毛利 `m`、每位被觸及者成本 `c`。
   - 門檻 τ 掃描：對 p≥τ 者發券，期望利潤 ≈ `TP×m − N_treated×c`。
   - 對照「全發券 / 都不發 / RFM 規則」三條線，找 profit-max 門檻 τ*。
   - **可觸及性版本**：只對「至少啟用一種行銷通路」者可發券，重畫一條（更貼實務）。
5. **可解釋性 → 行銷洞察**：SHAP 全域（beeswarm）+ top 特徵 dependence；邏輯迴歸係數表。轉成 **3–5 條可執行洞察**（例：t0 深折扣 + 單一低價品 → 多為路過；全價買核心品類 → 多為復活，應優先拉進會員方案）。

**輸出**：`output/B/figures/`（lift、gain、profit、calibration、SHAP）、`output/B/business_findings.md`

**驗收**：profit curve 顯示模型門檻策略的期望利潤 > 全發券與 RFM 規則。

### 需先確認的單位經濟假設（填進設定檔）
- `m`：一位復活者 90 天增量毛利 = 觀察到的復活者 `future_90d_revenue` 平均 × 毛利率（毛利率先用美妝保健常見區間，並標注為假設）。
- `c`：每位被觸及成本 = 折價券面額 × 兌換率，或直接給定行銷成本。
- 全部寫成可調參數，報告做敏感度（毛利率 / 兌換率變動時 τ* 怎麼變）。

---

## Sprint B5 — 穩健性、子群、報告組裝 ★ MUST(報告) / ◇(穩健性)

**步驟**
1. **子群報告** ★：分 `is_imputed`（True vs False）、分 `ShopId` 各報主要指標——展示模型在弱訊號子群（只買一次者）表現，是誠實度加分。
2. **穩健性檢驗** ◇：把復活定義改成「t0 後一個個人週期內再購」重跑，確認 top 特徵與 lift 結論不變（對「復活如何定義」不敏感 = 穩健）。
3. **漏斗圖** ★：沉睡客 → 回流（A 出口 = B 入口）→ 復活/路過，標人數與轉換率。
4. **NAPL 敘事** ★：把題目定位成「預測 Lost 客能否被拉回 Active」，對齊老師熟悉框架。
5. **限制與未來工作** ★：寫入「全動態滑動偵測」——B 因與 A 共用固定基準日母體而採靜態，但純訂單方法在邏輯上可推廣為全動態、擺脫單一時間節點、涵蓋季節性，這是行為資料（天生卡在 2023-09 起點）做不到的。此段即方法論貢獻，不需實作。
6. **核心賣點收束** ★：不需行為資料、純訂單即可預測復活機率；母體較行為版大；可即時於回購當下觸發決策；商業價值（省下被浪費的折扣 margin）> 模型分數。

**輸出**：`output/B/report_assets/`（圖表 + 洞察 + 漏斗 + 限制段）

---

## 建議資料夾結構

```
output/B/
├── b0_summary.json
├── event_table.parquet
├── features.parquet
├── models/
├── cv_results.json
├── figures/
├── business_findings.md
└── report_assets/
src/B/
├── b0_sanity.py / .sql
├── b1_label.py
├── b2_features.py
├── b3_train.py
├── b4_evaluate.py
└── b5_robustness_report.py
config/
└── unit_economics.yaml   # m, c, 毛利率, 兌換率（敏感度用）
```

---

## 三大致命風險（全程盯緊）

1. **Leakage**：每組特徵產出後跑斷言——確認沒有任何特徵取用 `OrderDateTime ≥ t0` 的資料（t0 訂單本身除外）。這是 B 唯一會致命的 bug。
2. **OOM**：`order_ts`(1.2GB) 一律 DuckDB 聚合到人/單層級再進 pandas。
3. **Censoring**：t0 ≤ 2023-12-01 一定要過濾乾淨，否則尾端觀察不滿 90 天的會被誤標路過。

---

## 執行優先序（時間被壓縮時的保命路線）

若工作時段被壓縮，至少完成：**B0 驗數 → B1 標籤 → B2 只做 ★ 特徵 → B3 邏輯迴歸 + LightGBM + 三基準 → B4 lift 圖 + profit curve + 校準 + 3 條洞察 → B5 漏斗 + 限制段。**
CatBoost、Optuna 調參、品類熵/趨勢等 ◇ 特徵、穩健性檢驗，全列 nice-to-have，有時間再補。
