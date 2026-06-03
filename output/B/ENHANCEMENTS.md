# B 組｜強化與調整紀錄（B0~B5 之後的追加工作）

> 本檔只收錄第一版 B0~B5 完成**之後**的額外分析與調整，不含原始 sprint 內容。
> 原始流程見 `B_FINAL_REPORT.md`（總報告）與各 sprint 報告。
> 共 5 項：任務 1~4 + baseline 改版。所有效能數字皆為**時間外 test**（train t0<2023-11、test t0≥2023-11），決策機率用校準後 LightGBM。

---

## 任務 1 — 修正版 Profit 分析（Uplift + Cannibalization）

**動機**：原版 B4 profit 假設「被觸及的復活者全貢獻全額增量毛利」，因 m/c≈11（毛利遠大於觸及成本）導致 profit-max 幾乎全發、模型僅勝全發券 1.5%，商業論述偏弱且過度樂觀。

**做法**：新增兩個更誠實的成本模型（原版保留對照），參數寫進 `config/unit_economics.yaml` 的 `revised_profit`。
- **模型 B（Uplift 參數化）**：引入 `u`=優惠券促成復活的比例，只有 u 比例的復活營收算增量。
- **模型 C（Cannibalization）**：對「被觸及且本來就會復活」者扣掉白送的折扣 margin。

**結果（模型 B，掃 u）**：

| u（真實 uplift） | 模型 τ* | profit-max 觸及比例 | 模型利潤 | 全發券利潤 | 模型 − 全發券 |
|----:|----:|------:|----------:|------------:|-----------:|
| 0.05 | 0.875 | **0.5%** | 965 | **−2,040,803** | +2,041,768 |
| 0.10 | 0.680 | **3.0%** | 41,125 | **−1,574,427** | +1,615,551 |
| 0.20 | 0.500 | **17.0%** | 317,079 | **−641,673** | +958,752 |
| 0.30 | 0.405 | 33.0% | 779,054 | 291,080 | +487,974 |
| 0.50 | 0.125 | 90.7% | 2,257,420 | 2,156,587 | +100,833 |

- **結論**：u ≤ 0.30 時模型轉為選擇性 targeting（觸及<80%）並大幅勝過全發券；**u ≤ 0.20 時全發券期望利潤為負**（觸及成本吃掉微薄增量）。在實務常見 10–30% uplift 區間，全發券會虧損，唯有模型只發給高分少數客才獲利。
- **模型 C（u_c=0.2）**：τ*=0.045、觸及 96.6%，僅勝全發券 +0.2%（cannibalization 罰則太輕，uplift 才是真正槓桿）。

**產出**：`B4/b_profit_revised.json`、`B4/figures/profit_uplift_vs_u.png`、`B4/figures/profit_cannibalization.png`
**腳本**：`src/B/b4b_profit_revised.py`

---

## 任務 2 — 品牌異質性表

**動機**：各 ShopId 基本率差異大（17%~52%），需量化「哪些品牌最值得用模型分群投放」。

**做法**：時間外 test 按 ShopId 拆 5 品牌，依 base rate 排序，每品牌算 n、base rate、ROC-AUC、PR-AUC、第1分位 lift、profit-max 觸及比例、模型 vs 全發券利潤差。profit 用任務 1 的 uplift 模型（u=0.2 代表值）。

**結果**：

| 品牌 | n | base rate | ROC-AUC | PR-AUC | 第1分位 lift | 每人利潤增益 |
|------|--:|:--------:|:------:|:-----:|:-----:|----:|
| **RZSHERLB** | 5,106 | **0.129** | 0.818 | 0.470 | **4.14×** | +31.1 |
| NOmceSRC | 2,425 | 0.231 | 0.589 | 0.295 | 1.40× | +29.6 |
| zXQPxhiL | 7,139 | 0.266 | 0.651 | 0.405 | 1.98× | +28.8 |
| hFwniXiB | 17,328 | 0.401 | 0.644 | 0.571 | 1.82× | +32.5 |
| 3WUOySTy | 6,574 | **0.485** | 0.631 | 0.607 | **1.44×** | +21.5 |

- **驗證「低復活率品牌 targeting 價值最大」成立**：base_rate vs lift 相關 **−0.70**、vs 每人利潤增益 **−0.59**。
- 最極端 RZSHERLB（復活率 12.9%）lift 高達 **4.14×**，最值得集中預算。
- **行動建議**：低 base rate 品牌（RZSHERLB、zXQPxhiL）優先導入模型分群；高 base rate 品牌（3WUOySTy）用簡單規則即可。

**產出**：`B5/report_assets/brand_heterogeneity.csv` / `.png` / `.json`
**腳本**：`src/B/b5b_brand_heterogeneity.py`

---

## 任務 3 — t0_amount 洞察佐證（並更正 B4 洞察 #2 方向）

**動機**：SHAP 顯示 t0_amount 為負向，但 B4 原洞察 #2 誤寫成「t0 金額越高越可能復活」。需以分桶佐證並更正。

**做法**：把 t0_amount 分 10 個 quantile 桶（全 144,919 母體），畫每桶復活率，並分 is_imputed True/False 各畫一次以控制子群混淆。

**結果**：
- **整體 Spearman = −0.90**：最低金額桶復活率 **41.9%** → 最高金額桶 **25.0%**。
- 控制 is_imputed 後趨勢仍成立（False −0.76、True −0.88，兩線不交叉）→ **非子群混淆造成的假象**。

> ⚠️ **更正**：B4 洞察 #2 原寫「較高 t0 金額者更可能復活」方向**錯誤**，正確為**負向**。

- **正確商業詮釋**：回購當下砸大錢者多為一次性「路過型」消費（囤貨、送禮、衝動），深折扣留存資源應留給中低額、更可能規律回購者。

**產出**：`B4/figures/t0_amount_revival.png`、`B4/t0_amount_bins.csv`
**腳本**：`src/B/b4c_t0amount_insight.py`

---

## 任務 4 — t0_payment_type 清理（資料品質澄清）

**動機**：B2 缺失率報表上 t0_payment_type「63.8% 缺失」看起來像資料品質問題，需釐清並清理。

**做法**：t0_payment_type 為 NaN 且該筆 t0 訂單通路為 POS → 補成 "POS"；其餘真缺失保留 NaN。重跑 LightGBM(raw+校準)看前後指標。

**結果**：
- 92,511 筆缺失中 **100% 為線下 POS 訂單**（店內無線上付款欄位），**真缺失 0 筆**。
- 補成 "POS" 後 LightGBM 時間外 test 的 ROC-AUC / PR-AUC / Brier **變動皆 0.0000**（樹模型只看類別分組、relabel 等價）。
- **意涵**：該「高缺失」可改述為「線下消費標記」這個有用特徵，而非缺陷。

**已持久化**：清理寫回 `features.parquet`、併入 `b2_features.py`、覆寫 `lightgbm.joblib` ×2、更新 test_predictions 的 p_lightgbm 兩欄（logreg/catboost 未動）。
**產出**：`B4/t0_paytype_cleanup.json`
**腳本**：`src/B/b3b_paytype_check.py`

---

## 任務 5 — Baseline 改版：RFM 規則 → RFM-only 邏輯迴歸

**動機**：原 B3 的「RFM 規則」是人工門檻（recency<中位數 + monetary>中位數），太弱、打贏它說服力不足。改成正經訓練的 RFM-only 邏輯迴歸，作為更公允、更難打的對照。

**做法**：只用 R/F/M 三個經典特徵（`hist_recency_days`、`hist_finish_count`、`hist_total_amount`）訓練邏輯迴歸（標準化 + 中位數補值 + class_weight=balanced），取代原 `baseline_rfm_rule`。

**結果（改版前後對照）**：

| Baseline | ROC-AUC | PR-AUC |
|----------|:-------:|:------:|
| 改版前（人工門檻規則 `baseline_rfm_rule`） | 0.562 | 0.392 |
| **改版後（`baseline_rfm_logreg`）** | **0.645** | **0.503** |

**完整對照（時間外 test）**：

| 模型 | ROC-AUC | PR-AUC |
|------|:-------:|:------:|
| baseline 多數類 | 0.500 | 0.344 |
| **baseline_rfm_logreg** | **0.645** | **0.503** |
| baseline NAPL 規則 | 0.570 | 0.379 |
| 邏輯迴歸（全特徵） | 0.706 | 0.559 |
| LightGBM | 0.710 | 0.569 |
| **CatBoost（最佳）** | **0.716** | **0.578** |

- **意涵**：主模型 CatBoost PR-AUC 0.578 vs RFM-logreg 0.503 → 領先 **+0.075（+15%）**。打贏一個正經 RFM 模型（0.50）比打贏人工規則（0.39）更能證明 **非 RFM 特徵（t0 訂單情境、會員等級、品牌、通路）帶來真實增量**。
- RFM-only 已達 PR-AUC 0.50，說明 R/F/M 本身就很有訊號；全特徵多出的 ~0.07 來自 B2 工程特徵。
- 驗收仍通過（主模型 > 全部三基準）。

**已更新**：`src/B/b3_train.py`（baseline 區塊 + 驗收引用）、`B3/cv_results.json`（`baseline_rfm_rule`→`baseline_rfm_logreg`）。重跑連帶刷新 models/ 與 test_predictions（資料/種子相同，與先前等價）。
**腳本**：`src/B/b3_train.py`

---

## 對既有報告的影響彙整

| 報告 | 受影響處 | 狀態 |
|------|---------|------|
| `B4/business_findings.md` | 附錄 A（任務 1/3/4）已附加；第 5 節洞察 #2 由任務 3 更正 | 已更新 |
| `B5/report_assets/B5_report.md` | 附錄 B（任務 2）已附加 | 已更新 |
| `B_FINAL_REPORT.md` | 已含四任務；**baseline 數字（任務 5）尚提及舊值 0.39，待同步** | ⚠️ 待同步 |
| `B3/cv_results.json` | baseline 名稱與數字已更新 | 已更新 |

> 註：多處報告正文仍寫「RFM 規則 PR-AUC≈0.39」，若要全面對齊新 baseline（0.50），需另行同步更新（可再交辦）。
