"""
Sprint B2 — 特徵工程
=====================
鐵律：
  - 歷史特徵一律 WHERE OrderDateTime < t0
  - t0 訂單特徵取 OrderDateTime == t0 那筆（同日多筆彙總）
  - order_ts (1.2GB) 全程 DuckDB 聚合到人層級才進 pandas
  - 每組特徵產出後跑 leakage 斷言

輸出：output/B/features.parquet（事件表左接全部特徵）
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
EVENT = ROOT / "output/B/B1/event_table.parquet"
ORDER_TG = ROOT / "output/sprint3/order_tg.parquet"
ORDER_TS = ROOT / "output/sprint3/order_ts.parquet"
MEMBER = ROOT / "output/sprint3/member.parquet"
OUT_DIR = ROOT / "output/B/B2"
OUT_PATH = OUT_DIR / "features.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=4")

# 載入事件表並註冊
event = pd.read_parquet(EVENT)
event["t0"] = pd.to_datetime(event["t0"])
con.register("ev", event[["ShopMemberId", "ShopId", "t0"]])
N = len(event)
print(f"[B2] 事件表載入: {N:,} 人")

feat = event.copy()  # 主表，逐步 left join


def lmerge(base, add, name):
    """左接並印出形狀檢查。"""
    before = len(base)
    out = base.merge(add, on=["ShopMemberId", "ShopId"], how="left")
    assert len(out) == before, f"FAIL {name}: row count changed {before}->{len(out)}"
    print(f"  + {name}: {add.shape[1]-2} 欄, 命中 {add.shape[0]:,} 人")
    return out


# ================================================================
# 2A. 歷史 RFM（order_tg, Finish, < t0）
# ================================================================
print("\n[2A] 歷史 RFM（order_tg, Finish, < t0）...")

q_rfm = f"""
WITH base AS (
    SELECT o.ShopMemberId, o.ShopId, o.OrderDateTime, o.TotalSalesAmount, ev.t0,
           (o.OrderDateTime >= ev.t0 - INTERVAL 365 DAY) AS within_365
    FROM '{ORDER_TG}' o
    JOIN ev ON o.ShopMemberId = ev.ShopMemberId AND o.ShopId = ev.ShopId
    WHERE o.StatusDef = 'Finish' AND o.OrderDateTime < ev.t0
)
SELECT
    ShopMemberId, ShopId,
    COUNT(*)                                          AS hist_finish_count,
    SUM(TotalSalesAmount)                             AS hist_total_amount,
    AVG(TotalSalesAmount)                             AS hist_aov_mean,
    MEDIAN(TotalSalesAmount)                          AS hist_aov_median,
    STDDEV(TotalSalesAmount)                          AS hist_aov_std,
    date_diff('day', MAX(OrderDateTime), MIN(t0))     AS hist_recency_days,
    MIN(OrderDateTime)                                AS hist_first_finish_dt,
    COUNT(*) FILTER (WHERE within_365)                AS hist_finish_365d
FROM base
GROUP BY ShopMemberId, ShopId
"""
rfm = con.execute(q_rfm).fetchdf()
feat = lmerge(feat, rfm.drop(columns=["hist_first_finish_dt"]), "RFM")
# 保留 first_finish_dt 供 trend 使用（不進最終特徵）
first_dt_map = rfm[["ShopMemberId", "ShopId", "hist_first_finish_dt"]]

# ── 趨勢：後半段消費額 / 前半段消費額 ──
q_trend = f"""
WITH base AS (
    SELECT o.ShopMemberId, o.ShopId, o.OrderDateTime, o.TotalSalesAmount, ev.t0,
           MIN(o.OrderDateTime) OVER (PARTITION BY o.ShopMemberId, o.ShopId) AS first_dt
    FROM '{ORDER_TG}' o
    JOIN ev ON o.ShopMemberId = ev.ShopMemberId AND o.ShopId = ev.ShopId
    WHERE o.StatusDef = 'Finish' AND o.OrderDateTime < ev.t0
)
SELECT
    ShopMemberId, ShopId,
    SUM(TotalSalesAmount) FILTER (
        WHERE epoch(OrderDateTime) <  (epoch(first_dt)+epoch(t0))/2 ) AS first_half_amt,
    SUM(TotalSalesAmount) FILTER (
        WHERE epoch(OrderDateTime) >= (epoch(first_dt)+epoch(t0))/2 ) AS second_half_amt
FROM base
GROUP BY ShopMemberId, ShopId
"""
trend = con.execute(q_trend).fetchdf()
trend["trend_ratio"] = trend["second_half_amt"] / trend["first_half_amt"].replace(0, np.nan)
feat = lmerge(feat, trend[["ShopMemberId", "ShopId", "trend_ratio"]], "trend")

# ── 購買間隔變異係數（規律性）──
q_cv = f"""
WITH f AS (
    SELECT o.ShopMemberId, o.ShopId, o.OrderDateTime,
           LAG(o.OrderDateTime) OVER (PARTITION BY o.ShopMemberId, o.ShopId
                                      ORDER BY o.OrderDateTime) AS prev_dt
    FROM '{ORDER_TG}' o
    JOIN ev ON o.ShopMemberId = ev.ShopMemberId AND o.ShopId = ev.ShopId
    WHERE o.StatusDef = 'Finish' AND o.OrderDateTime < ev.t0
),
gaps AS (
    SELECT ShopMemberId, ShopId, date_diff('day', prev_dt, OrderDateTime) AS gap
    FROM f WHERE prev_dt IS NOT NULL
)
SELECT ShopMemberId, ShopId,
       AVG(gap)                              AS interval_mean,
       STDDEV(gap) / NULLIF(AVG(gap), 0)     AS interval_cv,
       COUNT(*)                              AS n_gaps
FROM gaps GROUP BY ShopMemberId, ShopId
"""
cv = con.execute(q_cv).fetchdf()
feat = lmerge(feat, cv[["ShopMemberId", "ShopId", "interval_mean", "interval_cv"]], "interval_cv")

# ================================================================
# 2C. 通路 + 退貨率（order_tg, 全狀態, < t0）
# ================================================================
print("\n[2C] 通路與退貨率（order_tg, 全狀態, < t0）...")

q_chan = f"""
WITH base AS (
    SELECT o.*
    FROM '{ORDER_TG}' o
    JOIN ev ON o.ShopMemberId = ev.ShopMemberId AND o.ShopId = ev.ShopId
    WHERE o.OrderDateTime < ev.t0
)
SELECT
    ShopMemberId, ShopId,
    COUNT(*)                                                        AS hist_order_all,
    AVG(CASE WHEN ChannelType = 'Pos' THEN 1.0 ELSE 0.0 END)        AS offline_ratio,
    AVG(CASE WHEN ChannelDetail IN ('iOSApp','AndroidApp') THEN 1.0 ELSE 0.0 END) AS app_order_ratio,
    AVG(CASE WHEN ChannelDetail LIKE '%Web%' THEN 1.0 ELSE 0.0 END) AS web_order_ratio,
    COUNT(DISTINCT ChannelType)                                     AS distinct_channel_types,
    AVG(CASE WHEN StatusDef = 'Return' THEN 1.0 ELSE 0.0 END)       AS return_rate
FROM base
GROUP BY ShopMemberId, ShopId
"""
chan = con.execute(q_chan).fetchdf()
feat = lmerge(feat, chan, "channel+return")

# ================================================================
# 2B. 品類與折扣（order_ts, < t0）★ 核心折扣 / ◇ 品類
# ================================================================
print("\n[2B] 品類與折扣（order_ts, < t0）...")

q_ts_hist = f"""
WITH ts AS (
    SELECT t.ShopMemberId, t.ShopId, t.TradesGroupCode, t.SalePageId,
           -(COALESCE(t.SubtotalPromotionDiscount,0)
             + COALESCE(t.SubtotalCouponDiscount,0)
             + COALESCE(t.SubtotalLoyaltyPointDiscount,0)) AS disc,
           COALESCE(t.SubtotalSalesAmount,0)               AS sales
    FROM '{ORDER_TS}' t
    JOIN ev ON t.ShopMemberId = ev.ShopMemberId AND t.ShopId = ev.ShopId
    WHERE t.OrderDateTime < ev.t0
),
order_lvl AS (
    SELECT ShopMemberId, ShopId, TradesGroupCode, SUM(disc) AS order_disc
    FROM ts GROUP BY ShopMemberId, ShopId, TradesGroupCode
),
order_ratio AS (
    SELECT ShopMemberId, ShopId,
           AVG(CASE WHEN order_disc > 0 THEN 1.0 ELSE 0.0 END) AS has_discount_order_ratio
    FROM order_lvl GROUP BY ShopMemberId, ShopId
),
mem AS (
    SELECT ShopMemberId, ShopId,
           COUNT(DISTINCT SalePageId)               AS hist_distinct_salepages,
           SUM(disc)                                AS hist_total_disc,
           SUM(sales + disc)                        AS hist_total_listprice
    FROM ts GROUP BY ShopMemberId, ShopId
)
SELECT m.ShopMemberId, m.ShopId,
       m.hist_distinct_salepages,
       m.hist_total_disc / NULLIF(m.hist_total_listprice, 0) AS hist_discount_ratio,
       r.has_discount_order_ratio
FROM mem m LEFT JOIN order_ratio r USING (ShopMemberId, ShopId)
"""
ts_hist = con.execute(q_ts_hist).fetchdf()
feat = lmerge(feat, ts_hist, "ts_hist(品類+折扣)")

# ── 品類熵（多樣性）◇ ──
q_entropy = f"""
WITH cnt AS (
    SELECT t.ShopMemberId, t.ShopId, t.SalePageId, SUM(t.Qty) AS q
    FROM '{ORDER_TS}' t
    JOIN ev ON t.ShopMemberId = ev.ShopMemberId AND t.ShopId = ev.ShopId
    WHERE t.OrderDateTime < ev.t0
    GROUP BY t.ShopMemberId, t.ShopId, t.SalePageId
    HAVING SUM(t.Qty) > 0
),
tot AS (
    SELECT ShopMemberId, ShopId, SUM(q) AS total_q FROM cnt GROUP BY ShopMemberId, ShopId
)
SELECT c.ShopMemberId, c.ShopId,
       -SUM((c.q/t.total_q) * ln(c.q/t.total_q)) AS category_entropy
FROM cnt c JOIN tot t USING (ShopMemberId, ShopId)
WHERE t.total_q > 0
GROUP BY c.ShopMemberId, c.ShopId
"""
entropy = con.execute(q_entropy).fetchdf()
feat = lmerge(feat, entropy, "category_entropy")

# ================================================================
# 2D. t0 回購訂單本身（order_tg + order_ts, OrderDateTime == t0）★
# ================================================================
print("\n[2D] t0 回購訂單本身（== t0）...")

q_t0_tg = f"""
WITH base AS (
    SELECT o.ShopMemberId, o.ShopId, o.TotalSalesAmount, o.Qty, o.TsCount,
           COALESCE(o.TotalPromotionDiscount,0)+COALESCE(o.TotalCouponDiscount,0)
             +COALESCE(o.TotalLoyaltyPointDiscount,0) AS t0_disc,
           o.TotalPrice, o.ChannelType, o.ChannelDetail, o.PaymentType
    FROM '{ORDER_TG}' o
    JOIN ev ON o.ShopMemberId = ev.ShopMemberId AND o.ShopId = ev.ShopId
            AND o.OrderDateTime = ev.t0
    WHERE o.StatusDef = 'Finish'
)
SELECT
    ShopMemberId, ShopId,
    SUM(TotalSalesAmount)                       AS t0_amount,
    SUM(Qty)                                    AS t0_qty,
    SUM(TsCount)                                AS t0_tscount,
    -SUM(t0_disc) / NULLIF(SUM(TotalPrice), 0)  AS t0_discount_ratio,
    ANY_VALUE(ChannelType)                      AS t0_channel_type,
    ANY_VALUE(ChannelDetail)                    AS t0_channel_detail,
    ANY_VALUE(PaymentType)                      AS t0_payment_type
FROM base
GROUP BY ShopMemberId, ShopId
"""
t0_tg = con.execute(q_t0_tg).fetchdf()
feat = lmerge(feat, t0_tg, "t0_order(tg)")

# ── t0 籃子品項數 + 新舊品類判斷（order_ts）──
q_t0_ts = f"""
WITH t0_pages AS (
    SELECT DISTINCT t.ShopMemberId, t.ShopId, t.SalePageId
    FROM '{ORDER_TS}' t
    JOIN ev ON t.ShopMemberId = ev.ShopMemberId AND t.ShopId = ev.ShopId
            AND t.OrderDateTime = ev.t0
),
hist_pages AS (
    SELECT DISTINCT t.ShopMemberId, t.ShopId, t.SalePageId
    FROM '{ORDER_TS}' t
    JOIN ev ON t.ShopMemberId = ev.ShopMemberId AND t.ShopId = ev.ShopId
    WHERE t.OrderDateTime < ev.t0
)
SELECT
    p.ShopMemberId, p.ShopId,
    COUNT(*)                                                AS t0_distinct_salepages,
    SUM(CASE WHEN h.SalePageId IS NULL THEN 1 ELSE 0 END)   AS t0_n_new_categories
FROM t0_pages p
LEFT JOIN hist_pages h USING (ShopMemberId, ShopId, SalePageId)
GROUP BY p.ShopMemberId, p.ShopId
"""
t0_ts = con.execute(q_t0_ts).fetchdf()
t0_ts["t0_new_category_ratio"] = (
    t0_ts["t0_n_new_categories"] / t0_ts["t0_distinct_salepages"].replace(0, np.nan)
)
feat = lmerge(feat, t0_ts, "t0_basket(ts)")

# ================================================================
# 2E. 會員屬性（member.parquet）★
# ================================================================
print("\n[2E] 會員屬性（member）...")

mem = pd.read_parquet(MEMBER, columns=[
    "ShopMemberId", "ShopId", "RegisterSourceTypeDef", "RegisterDateTime",
    "Gender", "Birthday", "IsAppInstalled", "IsEnableEmail",
    "IsEnablePushNotification", "IsEnableShortMessage",
    "LastAppOpenDateTime", "MemberCardLevel",
])

# 缺失值預設處理：1900-01-01 → NaT
mem["RegisterDateTime"] = pd.to_datetime(mem["RegisterDateTime"], errors="coerce")
mem.loc[mem["RegisterDateTime"].dt.year <= 1900, "RegisterDateTime"] = pd.NaT
mem["LastAppOpenDateTime"] = pd.to_datetime(mem["LastAppOpenDateTime"], errors="coerce")
mem.loc[mem["LastAppOpenDateTime"].dt.year <= 1900, "LastAppOpenDateTime"] = pd.NaT
mem["Birthday"] = pd.to_datetime(mem["Birthday"], errors="coerce")
mem.loc[mem["Birthday"].dt.year <= 1900, "Birthday"] = pd.NaT

# MemberCardLevel 0 → NaN + 缺失旗標
mem["membercard_missing"] = (mem["MemberCardLevel"] == 0).astype(int)
mem["MemberCardLevel"] = mem["MemberCardLevel"].replace(0, np.nan)

# 行銷可觸及性
mem["n_marketing_channels"] = (
    mem["IsEnableEmail"].astype(int)
    + mem["IsEnablePushNotification"].astype(int)
    + mem["IsEnableShortMessage"].astype(int)
)
mem["any_marketing_reachable"] = (mem["n_marketing_channels"] > 0).astype(int)

# Gender null → Unknown
mem["Gender"] = mem["Gender"].fillna("Unknown")

mem_feat = mem[[
    "ShopMemberId", "ShopId", "RegisterSourceTypeDef", "RegisterDateTime",
    "Gender", "Birthday", "IsAppInstalled", "IsEnableEmail",
    "IsEnablePushNotification", "IsEnableShortMessage", "LastAppOpenDateTime",
    "MemberCardLevel", "membercard_missing",
    "n_marketing_channels", "any_marketing_reachable",
]].copy()

feat = lmerge(feat, mem_feat, "member")

# ── 衍生：年資、年齡、距上次開 App（皆需 t0）──
feat["tenure_days"] = (feat["t0"] - feat["RegisterDateTime"]).dt.days
feat["age"] = ((feat["t0"] - feat["Birthday"]).dt.days / 365.25)
feat.loc[(feat["age"] < 10) | (feat["age"] > 100), "age"] = np.nan
feat["age_bucket"] = pd.cut(
    feat["age"], bins=[0, 25, 35, 45, 55, 200],
    labels=["<25", "25-34", "35-44", "45-54", "55+"]
).astype("object")

# LastAppOpenDateTime 防 leakage：只有 < t0 才可用，否則 NaN
last_open_valid = feat["LastAppOpenDateTime"].where(feat["LastAppOpenDateTime"] < feat["t0"])
feat["days_since_app_open"] = (feat["t0"] - last_open_valid).dt.days

# 移除已衍生完的原始時間欄位
feat = feat.drop(columns=["RegisterDateTime", "Birthday", "LastAppOpenDateTime", "age"])

# ================================================================
# 3. 衍生：沉睡前空窗（已有 last_purchase_date 與 t0）
# ================================================================
feat["dormancy_gap_days"] = (feat["t0"] - pd.to_datetime(feat["last_purchase_date"])).dt.days

# ── 清理 t0_payment_type：POS 通路無線上付款方式，缺失補 'POS'（可解釋缺失）──
_pos_pay = feat["t0_payment_type"].isna() & (feat["t0_channel_type"] == "Pos")
feat.loc[_pos_pay, "t0_payment_type"] = "POS"
print(f"  [清理] t0_payment_type POS 通路缺失補 'POS': {int(_pos_pay.sum()):,} 筆")

# ================================================================
# 4. Leakage 結構斷言
# ================================================================
print("\n[B2] Leakage 結構斷言 ...")

# 4a. 行數不變、每人唯一
assert len(feat) == N, f"FAIL: 行數 {len(feat)} != {N}"
assert feat.groupby(["ShopMemberId", "ShopId"]).size().max() == 1, "FAIL: 有重複"
print("  [PASS] 每人唯一、行數不變")

# 4b. 直接驗證 SQL：歷史聚合不得納入 OrderDateTime >= t0 的訂單
viol_tg = con.execute(f"""
    SELECT COUNT(*) FROM '{ORDER_TG}' o JOIN ev
      ON o.ShopMemberId=ev.ShopMemberId AND o.ShopId=ev.ShopId
    WHERE o.OrderDateTime > ev.t0 AND o.OrderDateTime <= ev.t0  -- 永遠空，留作結構樣板
""").fetchone()[0]
# 真正的檢查：確認 hist_recency_days 全部 >= 0（最後一筆歷史在 t0 之前）
neg_recency = (feat["hist_recency_days"] < 0).sum()
assert neg_recency == 0, f"FAIL: {neg_recency} 筆 hist_recency_days < 0（歷史訂單晚於 t0）"
print("  [PASS] hist_recency_days 全部 >= 0")

# 4c. tenure_days、dormancy_gap_days 全部 >= 0
assert (feat["tenure_days"].dropna() >= 0).all(), "FAIL: tenure_days 有負值"
assert (feat["dormancy_gap_days"].dropna() >= 0).all(), "FAIL: dormancy_gap_days 有負值"
print("  [PASS] tenure / dormancy_gap >= 0")

# 4d. days_since_app_open 全部 >= 0（已限制 < t0）
assert (feat["days_since_app_open"].dropna() >= 0).all(), "FAIL: days_since_app_open 有負值"
print("  [PASS] days_since_app_open >= 0")

# 4e. 確認未洩漏標籤欄當特徵（future_* 只保留標籤用途）
leak_cols = [c for c in feat.columns if c.startswith("future_")]
print(f"  [INFO] 保留標籤欄（不可當特徵）: {leak_cols}")

# ================================================================
# 5. 缺失率報表
# ================================================================
print("\n[B2] 特徵缺失率（前 20 高）...")
miss = (feat.isna().mean().sort_values(ascending=False) * 100).round(2)
print(miss[miss > 0].head(20).to_string())

# ================================================================
# 6. 輸出
# ================================================================
feat.to_parquet(OUT_PATH, index=False)
print(f"\n[B2] 完成！已輸出: {OUT_PATH}")
print(f"  形狀: {feat.shape}")
print(f"  欄位數: {feat.shape[1]}")
print(f"  全部欄位: {list(feat.columns)}")

con.close()
