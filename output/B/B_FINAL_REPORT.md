# B 組總報告｜沉睡客「復活 vs 路過」預測

> 單頁完整版：串接 Sprint B0~B5 與四項後續強化任務。
> 定位：A→B 兩階段喚回漏斗的下游。A 預測「沉睡客會不會回流」，B 預測「回流者會復活還是路過」。
> 決策模型：校準後 LightGBM（isotonic）；所有效能數字來自**時間外 test**（train t0<2023-11、test t0≥2023-11）。

---

## 0. 核心定義（全 sprint 共用）

| 項目 | 定義 |
|------|------|
| 沉睡客母體 | 沿用 A 組 `dormant_members.parquet`（2023-09-01 靜態基準，1,112,087 人） |
| 回流 t0 | 該沉睡客 **OrderDateTime ≥ 2023-09-01 且 StatusDef=Finish** 的第一筆訂單時間 |
| B 母體 | 有 t0 且 **t0 ≤ 2023-12-01**（確保完整 90 天觀察窗，資料止於 2024-02-29） |
| 標籤 `revived` | `(t0, t0+90天]` 內有 ≥1 筆 Finish → 復活(1)；否則 → 路過(0) |
| Leakage 鐵律 | 特徵只能用 OrderDateTime < t0 的歷史 + t0 那筆訂單本身 |
| 建模單位 | 「人」（ShopMemberId 不跨 ShopId）；五品牌 pooled、ShopId 當特徵 |

**三大致命風險全程盯緊**：Leakage（特徵 < t0）、OOM（order_ts 1.2GB 一律 DuckDB）、Censoring（t0 ≤ 2023-12-01）。

---

## 1. B0 — 母體驗數

| 層級 | 人數 | 備註 |
|------|-----:|------|
| 沉睡客母體 | 1,112,087 | A 組出口 |
| 有 t0（首筆 Finish ≥ 2023-09-01） | 232,328 | 20.9% of 沉睡 |
| **B 母體（t0 ≤ 2023-12-01）** | **144,919** | 可建模 |
| → 復活 (1) | 54,475 | **基本率 37.6%** |
| → 路過 (0) | 90,444 | 62.4% |

- `order_tg` 時間範圍 2022-01 ~ 2024-02，歷史回看 >1.5 年，特徵工程充足。
- 復活率 37.6% 不極端，少數類 54,475 遠超門檻，**不需 resample**。
- 子群基本率差異大：is_imputed=False 46.2% / True 25.6%；ShopId 17%~52%。

**樣本充足性全 PASS**：總量 >10k、少數類 >1k、EPV≈1362、各子群數千以上。

---

## 2. B1 — 事件表與標籤

- 輸出 `event_table.parquet`（144,919，一人一列），含 `revived`、`future_90d_revenue`（標籤用、不可當特徵）。
- Leakage 斷言全過：每人唯一、revived 無缺值、t0 全 ≤ 2023-12-01、last_purchase_date < t0。

---

## 3. B2 — 特徵工程（54 欄）

每組特徵產出後跑 leakage 斷言（hist_recency/tenure/dormancy_gap/days_since_app_open 全 ≥ 0）。

| 群組 | 代表特徵 |
|------|---------|
| 歷史 RFM（<t0, Finish） | hist_finish_count、hist_total_amount、AOV(mean/median/std)、hist_recency_days、hist_finish_365d、trend_ratio、interval_cv |
| 品類與折扣（<t0, order_ts） | hist_discount_ratio★、has_discount_order_ratio★、hist_distinct_salepages、category_entropy |
| 通路與退貨（<t0） | offline_ratio、app/web_order_ratio、distinct_channel_types、return_rate |
| **t0 訂單本身**（==t0）★最強 | t0_amount、t0_qty、t0_tscount、t0_discount_ratio★、t0_channel_type/detail、t0_payment_type、t0_new_category_ratio |
| 會員屬性 | MemberCardLevel、n_marketing_channels★、any_marketing_reachable★、tenure_days、age_bucket、days_since_app_open、dormancy_gap_days |

> 缺失多為結構性合理（只買一次者無購買間隔）。樹模型原生吃 NaN；logreg 補值 + 缺失旗標。

---

## 4. B3 — 建模（時間外驗證）

| 模型 | ROC-AUC | PR-AUC | Brier |
|------|:-------:|:------:|:-----:|
| baseline 多數類 | 0.500 | 0.344 | 0.228 |
| baseline **RFM-only 邏輯迴歸** | 0.645 | 0.503 | 0.221 |
| baseline NAPL 規則 | 0.570 | 0.379 | 0.294 |
| 邏輯迴歸 | 0.706 | 0.559 | 0.206 |
| LightGBM | 0.710 | 0.569 | 0.203 |
| **CatBoost（最佳）** | **0.716** | **0.578** | 0.202 |
| LightGBM 校準版 | 0.708 | 0.554 | **0.197** |

- **驗收通過**：主模型 PR-AUC 0.578 vs 最強基準 RFM-logreg 0.503，勝出 **+15%**（見 `CANONICAL_NUMBERS.md`）。
- 校準有效（Brier 0.203→0.197），機率可直接用於期望利潤決策。
- 隨機切分 PR-AUC 0.626 > 時間外 0.569，反映時間漂移（11 月後復活率下降），時間外為誠實估計。

---

## 5. B4 — 評估與商業價值

- **辨識力**：ROC-AUC 0.708、PR-AUC 0.554。
- **排序**：Top 20% 名單涵蓋 **34.9% 復活者、43.4% 未來復活營收**；第 1 decile lift ≈ 2.0×。
- **SHAP 全域 Top**：ShopId、MemberCardLevel、t0_channel_type、offline_ratio、hist_finish_count、t0_channel_detail。
  - MemberCardLevel 高 → 強烈正向；offline_ratio/hist_finish_count 高 → 正向；**t0_amount 高 → 負向**（見任務 3）。
- **原版 Profit**（m=704, c=65，m/c≈11）：profit-max τ*=0.12、觸及 92%，模型僅勝全發券 +1.5%。
  - ⚠️ 此為早期「全額毛利」框架、過度樂觀，**已被 Block 1 uplift 修正版取代為主敘事**。最終經濟結論（單位修正後）：實務 uplift 比例 15~25% 下全發券落損益兩平微虧側、模型多賺 NT$0.7~1.3M，詳見 `B4/economics/ECONOMIC_NARRATIVE.md` 與 `CANONICAL_NUMBERS.md`。

---

## 6. B5 — 穩健性、子群、漏斗

**子群報告**（時間外 test）：

| 子群 | n | base rate | ROC-AUC | PR-AUC | decile1 lift |
|------|--:|:--------:|:------:|:-----:|:-----:|
| overall | 38,572 | 0.344 | 0.708 | 0.554 | 2.00× |
| is_imputed=False | 20,559 | 0.422 | 0.684 | 0.600 | 1.70× |
| is_imputed=True | 18,013 | 0.255 | 0.679 | 0.420 | 1.93× |

**穩健性**：復活定義改「t0 後一個個人週期內再購」重跑，樣本一致率 91.3%、Top-10 特徵重疊 7/10 → 結論不依賴單一窗口。

**漏斗**：沉睡 1,112,087 → 回流下單 232,328 → 可建模 144,919 → 復活 54,475 / 路過 90,444。

**限制與賣點**：靜態母體（可推廣為全動態滑動偵測，方法論貢獻）；純訂單即可預測、母體較行為版大、可即時於回購當下觸發；商業價值 > 模型分數。

---

# 後續強化任務（第一版完成後追加）

## 任務 1 — 修正版 Profit（Uplift + Cannibalization）★關鍵

原版假設「被觸及的復活者全貢獻全額增量毛利」過於樂觀。引入 `u`=優惠券促成復活的比例：

| u（真實 uplift） | 模型 τ* | profit-max 觸及比例 | 模型利潤 | 全發券利潤 | 差距 |
|----:|----:|------:|----------:|------------:|-----------:|
| 0.05 | 0.875 | **0.5%** | 965 | **−2,040,803** | +2,041,768 |
| 0.10 | 0.680 | **3.0%** | 41,125 | **−1,574,427** | +1,615,551 |
| 0.20 | 0.500 | **17.0%** | 317,079 | **−641,673** | +958,752 |
| 0.30 | 0.405 | 33.0% | 779,054 | 291,080 | +487,974 |
| 0.50 | 0.125 | 90.7% | 2,257,420 | 2,156,587 | +100,833 |

**結論**：u ≤ 0.30 時模型轉為選擇性 targeting（觸及<80%）並大幅勝過全發券；**u ≤ 0.20 時全發券期望利潤為負**。在實務常見 10–30% uplift 區間，全發券會虧損，唯有模型只發給高分少數客才獲利。
Cannibalization 版（u_c=0.2）：τ*=0.045、僅勝全發券 +0.2%（罰則太輕，uplift 才是真正槓桿）。

## 任務 2 — 品牌異質性表

依 base rate 排序，驗證「低復活率品牌 targeting 價值最大」：

| 品牌 | n | base rate | ROC-AUC | PR-AUC | decile1 lift | 每人利潤增益 |
|------|--:|:--------:|:------:|:-----:|:-----:|----:|
| **RZSHERLB** | 5,106 | **0.129** | 0.818 | 0.470 | **4.14×** | +31.1 |
| NOmceSRC | 2,425 | 0.231 | 0.589 | 0.295 | 1.40× | +29.6 |
| zXQPxhiL | 7,139 | 0.266 | 0.651 | 0.405 | 1.98× | +28.8 |
| hFwniXiB | 17,328 | 0.401 | 0.644 | 0.571 | 1.82× | +32.5 |
| 3WUOySTy | 6,574 | **0.485** | 0.631 | 0.607 | **1.44×** | +21.5 |

base_rate vs lift 相關 **−0.70**、vs 每人利潤增益 **−0.59**。**最低復活率品牌（RZSHERLB 12.9%）lift 高達 4.14×**，最值得集中預算。

## 任務 3 — t0_amount 洞察佐證（更正 B4 洞察 #2）

> ⚠️ 更正：B4 原洞察 #2「t0 金額越高越可能復活」方向**有誤**。正確為**負向**。

t0_amount 分 10 桶（全母體）：**整體 Spearman = −0.90**（最低桶 41.9% → 最高桶 25.0%）。控制 is_imputed 後趨勢仍成立（False −0.76、True −0.88，兩線不交叉）→ 非子群混淆。
**詮釋**：回購當下砸大錢者多為一次性「路過型」消費；深折扣留存資源應留給中低額、更可能規律回購者。

## 任務 4 — t0_payment_type 清理（資料品質澄清）

「63.8% 缺失」的 92,511 筆**全部為線下 POS 訂單**（店內無線上付款欄位），真缺失 0 筆。補成 "POS" 後重訓 LightGBM(raw+校準)，時間外 test 指標**變動皆 0.0000**（樹模型只看分組、relabel 等價）。該欄可改述為「線下消費標記」這個有用特徵，而非缺陷。

---

## 7. 腳本執行說明（src/B/）

```
# 主流程（依序）— B2 含 1.2GB order_ts，務必用 venv
./venv/Scripts/python src/B/b0_sanity.py            # → output/B/B0/
./venv/Scripts/python src/B/b1_label.py             # → output/B/B1/
./venv/Scripts/python src/B/b2_features.py          # → output/B/B2/  (已含任務4清理)
./venv/Scripts/python src/B/b3_train.py             # → output/B/B3/
./venv/Scripts/python src/B/b4_evaluate.py          # → output/B/B4/ (原版 profit + SHAP)
./venv/Scripts/python src/B/b5_robustness_report.py # → output/B/B5/

# 強化任務（B2/B3 完成後可獨立執行）
./venv/Scripts/python src/B/b4b_profit_revised.py    # 任務1 修正 profit → B4/
./venv/Scripts/python src/B/b5b_brand_heterogeneity.py # 任務2 品牌異質性 → B5/report_assets/
./venv/Scripts/python src/B/b4c_t0amount_insight.py  # 任務3 t0_amount 佐證 → B4/
./venv/Scripts/python src/B/b3b_paytype_check.py     # 任務4 paytype 清理驗證 → B4/
```

設定檔：`config/unit_economics.yaml`（毛利率、券面額、兌換率、uplift_grid、敏感度範圍）。

## 8. 產出位置速查

| 想找 | 路徑 |
|------|------|
| 母體/基本率 | `B0/b0_summary.json` |
| 特徵表 | `B2/features.parquet` |
| 模型檔 | `B3/models/` |
| 模型比較 | `B3/cv_results.json` |
| 商業洞察（含附錄 A：任務1/3/4） | `B4/business_findings.md` |
| 評估圖（ROC/lift/校準/profit/SHAP/t0_amount） | `B4/figures/` |
| 穩健性/子群/漏斗/限制（含附錄 B：任務2） | `B5/report_assets/B5_report.md` |
| 品牌異質性 | `B5/report_assets/brand_heterogeneity.*` |
| 資產索引 | `B5/report_assets/INDEX.md` |
| 資料夾結構說明 | `output/B/README.md` |
