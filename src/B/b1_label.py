"""
Sprint B1 — 事件表與標籤
=========================
產出 B 主表（一人一列），含 revived 標籤與 future_90d_revenue。
"""

import duckdb
import pandas as pd
from pathlib import Path

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
DORMANT = ROOT / "output/sprint1/dormant_members.parquet"
ORDER_TG = ROOT / "output/sprint3/order_tg.parquet"
MEMBER = ROOT / "output/sprint3/member.parquet"
OUT_DIR = ROOT / "output/B/B1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "event_table.parquet"

con = duckdb.connect()

# ================================================================
# 1. 計算每人 t0（第一筆 Finish >= 2023-09-01）
# ================================================================
print("[B1] Step 1: 計算 t0 ...")

t0_df = con.execute(f"""
    SELECT
        ShopMemberId,
        ShopId,
        MIN(OrderDateTime) AS t0
    FROM '{ORDER_TG}'
    WHERE StatusDef = 'Finish'
      AND OrderDateTime >= '2023-09-01'
    GROUP BY ShopMemberId, ShopId
""").fetchdf()

print(f"  有 t0 的人: {len(t0_df):,}")

# ================================================================
# 2. 內接 dormant_members（只保留沉睡客）
# ================================================================
print("[B1] Step 2: 內接 dormant_members ...")

dormant = pd.read_parquet(DORMANT)
t0_merged = t0_df.merge(
    dormant[["ShopMemberId", "is_imputed", "personal_cycle_days",
             "last_purchase_date", "purchase_count"]],
    on="ShopMemberId",
    how="inner"
)

# ================================================================
# 3. 過濾 t0 <= 2023-12-01（censoring 安全線）
# ================================================================
print("[B1] Step 3: 過濾 t0 <= 2023-12-01 ...")

t0_merged["t0"] = pd.to_datetime(t0_merged["t0"])
usable = t0_merged[t0_merged["t0"] <= "2023-12-01"].copy()
print(f"  B 母體: {len(usable):,}")

# ================================================================
# 4. 貼標：(t0, t0+90天] 內有 Finish → revived=1
#    同時算 future_90d_revenue（給 B4 profit curve 用）
# ================================================================
print("[B1] Step 4: 貼標 revived + future_90d_revenue ...")

con.register("usable", usable[["ShopMemberId", "ShopId", "t0"]])

label_df = con.execute(f"""
    WITH future_orders AS (
        SELECT
            u.ShopMemberId,
            u.ShopId,
            u.t0,
            o.TotalSalesAmount
        FROM usable u
        JOIN '{ORDER_TG}' o
          ON u.ShopMemberId = o.ShopMemberId
         AND u.ShopId = o.ShopId
        WHERE o.StatusDef = 'Finish'
          AND o.OrderDateTime > u.t0
          AND o.OrderDateTime <= u.t0 + INTERVAL 90 DAY
    )
    SELECT
        ShopMemberId,
        ShopId,
        t0,
        COUNT(*)              AS future_finish_count,
        SUM(TotalSalesAmount) AS future_90d_revenue
    FROM future_orders
    GROUP BY ShopMemberId, ShopId, t0
""").fetchdf()

usable = usable.merge(
    label_df[["ShopMemberId", "ShopId", "future_finish_count", "future_90d_revenue"]],
    on=["ShopMemberId", "ShopId"],
    how="left"
)
usable["future_finish_count"] = usable["future_finish_count"].fillna(0).astype(int)
usable["future_90d_revenue"] = usable["future_90d_revenue"].fillna(0.0)
usable["revived"] = (usable["future_finish_count"] >= 1).astype(int)

# ================================================================
# 5. 整理最終欄位
# ================================================================
event_table = usable[[
    "ShopMemberId", "ShopId", "t0",
    "is_imputed", "personal_cycle_days", "last_purchase_date", "purchase_count",
    "revived", "future_finish_count", "future_90d_revenue"
]].copy()

event_table["last_purchase_date"] = pd.to_datetime(event_table["last_purchase_date"])

# ================================================================
# 6. Leakage 防呆斷言
# ================================================================
print("[B1] Step 6: Leakage 防呆斷言 ...")

# 6a. 每人唯一
assert event_table.groupby(["ShopMemberId", "ShopId"]).size().max() == 1, \
    "FAIL: 有人重複出現！"
print("  [PASS] 每人唯一")

# 6b. revived 無缺值
assert event_table["revived"].isna().sum() == 0, \
    "FAIL: revived 有缺值！"
print("  [PASS] revived 無缺值")

# 6c. t0 全部 <= 2023-12-01
assert (event_table["t0"] <= "2023-12-01").all(), \
    "FAIL: 有 t0 > 2023-12-01！"
print("  [PASS] t0 全部 <= 2023-12-01")

# 6d. 事件表不含任何衍生自 t0 之後的欄位（除 future_* 標籤）
allowed_future_cols = {"revived", "future_finish_count", "future_90d_revenue"}
all_cols = set(event_table.columns)
safe_cols = {"ShopMemberId", "ShopId", "t0",
             "is_imputed", "personal_cycle_days", "last_purchase_date", "purchase_count"}
unexpected = all_cols - safe_cols - allowed_future_cols
assert len(unexpected) == 0, f"FAIL: 發現未預期欄位 {unexpected}"
print("  [PASS] 無未預期欄位")

# 6e. last_purchase_date < t0（沉睡客的最後購買日必在 t0 之前）
invalid_dates = event_table[event_table["last_purchase_date"] >= event_table["t0"]]
if len(invalid_dates) > 0:
    print(f"  [WARN] {len(invalid_dates)} 筆 last_purchase_date >= t0，檢視中...")
    # 這些可能是 last_purchase_date 和 t0 同日（沉睡前最後一筆就是 t0）
    # 但根據定義 t0 是第一筆 Finish >= 2023-09-01，而 last_purchase_date
    # 是沉睡判定時的最後購買日，應在基準日之前
    same_day = event_table[event_table["last_purchase_date"] == event_table["t0"].dt.normalize()]
    print(f"    其中同日: {len(same_day)} 筆")
else:
    print("  [PASS] last_purchase_date 全部 < t0")

print()

# ================================================================
# 7. 輸出
# ================================================================
event_table.to_parquet(OUT_PATH, index=False)
print(f"[B1] 完成！已輸出: {OUT_PATH}")
print(f"  筆數: {len(event_table):,}")
print(f"  欄位: {list(event_table.columns)}")
print(f"  revived 分佈:\n{event_table['revived'].value_counts().to_string()}")
print(f"  dtypes:\n{event_table.dtypes.to_string()}")

con.close()
