"""
Sprint B0 — 資料健檢與母體驗數
=================================
確認 B 母體規模、資料時間範圍、基本率（復活 vs 路過），
並分 is_imputed / ShopId 子群檢視。
"""

import json
import duckdb
import pandas as pd
from pathlib import Path

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
DORMANT = ROOT / "output/sprint1/dormant_members.parquet"
ORDER_TG = ROOT / "output/sprint3/order_tg.parquet"
MEMBER = ROOT / "output/sprint3/member.parquet"
OUT_DIR = ROOT / "output/B/B0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# ================================================================
# 1. order_tg 時間範圍
# ================================================================
time_range = con.execute(f"""
    SELECT
        MIN(OrderDateTime) AS order_min,
        MAX(OrderDateTime) AS order_max,
        COUNT(*)           AS total_rows,
        COUNT(DISTINCT ShopMemberId) AS distinct_members
    FROM '{ORDER_TG}'
""").fetchdf()

print("=== order_tg 時間範圍 ===")
print(time_range.to_string(index=False))
print()

# ================================================================
# 2. 計算每人 t0（第一筆 Finish 訂單 >= 2023-09-01）
# ================================================================
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

print(f"=== 有 t0 的沉睡客候選（Finish >= 2023-09-01）=== {len(t0_df):,} 人")
print(f"    t0 範圍: {t0_df['t0'].min()} ~ {t0_df['t0'].max()}")
print()

# ================================================================
# 3. 左接 dormant_members，只保留沉睡客
# ================================================================
dormant = pd.read_parquet(DORMANT)
print(f"=== dormant_members 筆數: {len(dormant):,} ===")
print(f"    欄位: {list(dormant.columns)}")
print()

# merge：只保留在沉睡客名單中的人
t0_merged = t0_df.merge(
    dormant[["ShopMemberId", "is_imputed", "personal_cycle_days",
             "last_purchase_date", "purchase_count"]],
    on="ShopMemberId",
    how="inner"
)
print(f"=== 有 t0 且為沉睡客: {len(t0_merged):,} 人 ===")
print()

# ================================================================
# 4. 過濾 t0 <= 2023-12-01（censoring 安全線）
# ================================================================
t0_merged["t0"] = pd.to_datetime(t0_merged["t0"])
usable = t0_merged[t0_merged["t0"] <= "2023-12-01"].copy()
n_usable = len(usable)
print(f"=== B 母體（t0 <= 2023-12-01）: {n_usable:,} 人 ===")
print()

# ================================================================
# 5. 貼標：(t0, t0 + 90天] 內有 Finish → revived=1
# ================================================================
# 把 usable 註冊進 DuckDB 做 join
con.register("usable", usable[["ShopMemberId", "ShopId", "t0"]])

label_df = con.execute(f"""
    WITH future_orders AS (
        SELECT
            u.ShopMemberId,
            u.ShopId,
            u.t0,
            o.OrderDateTime,
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

# merge back
usable = usable.merge(
    label_df[["ShopMemberId", "ShopId", "future_finish_count", "future_90d_revenue"]],
    on=["ShopMemberId", "ShopId"],
    how="left"
)
usable["future_finish_count"] = usable["future_finish_count"].fillna(0).astype(int)
usable["future_90d_revenue"] = usable["future_90d_revenue"].fillna(0.0)
usable["revived"] = (usable["future_finish_count"] >= 1).astype(int)

n_revived = usable["revived"].sum()
n_passerby = n_usable - n_revived
base_rate = n_revived / n_usable

print("=" * 60)
print("  B 母體基本率")
print("=" * 60)
print(f"  總人數:   {n_usable:>10,}")
print(f"  復活(1):  {n_revived:>10,}  ({base_rate:.2%})")
print(f"  路過(0):  {n_passerby:>10,}  ({1-base_rate:.2%})")
print()

# ================================================================
# 6. 分 is_imputed 子群
# ================================================================
print("=== 分 is_imputed 子群 ===")
imp_group = usable.groupby("is_imputed").agg(
    n=("revived", "size"),
    n_revived=("revived", "sum"),
).reset_index()
imp_group["base_rate"] = imp_group["n_revived"] / imp_group["n"]
imp_group["pct_of_total"] = imp_group["n"] / n_usable
print(imp_group.to_string(index=False))
print()

# ================================================================
# 7. 分 ShopId 子群
# ================================================================
print("=== 分 ShopId 子群 ===")
shop_group = usable.groupby("ShopId").agg(
    n=("revived", "size"),
    n_revived=("revived", "sum"),
).reset_index()
shop_group["base_rate"] = shop_group["n_revived"] / shop_group["n"]
shop_group["pct_of_total"] = shop_group["n"] / n_usable
print(shop_group.to_string(index=False))
print()

# ================================================================
# 8. t0 月份分佈
# ================================================================
print("=== t0 月份分佈 ===")
usable["t0_month"] = usable["t0"].dt.to_period("M").astype(str)
month_dist = usable.groupby("t0_month").agg(
    n=("revived", "size"),
    n_revived=("revived", "sum"),
).reset_index()
month_dist["base_rate"] = month_dist["n_revived"] / month_dist["n"]
print(month_dist.to_string(index=False))
print()

# ================================================================
# 9. 復活者 future_90d_revenue 分佈
# ================================================================
revived_only = usable[usable["revived"] == 1]
rev_stats = revived_only["future_90d_revenue"].describe()
print("=== 復活者 future_90d_revenue 分佈 ===")
print(rev_stats)
print()

# ================================================================
# 10. 樣本量充足性評估
# ================================================================
print("=" * 60)
print("  樣本量充足性評估")
print("=" * 60)

min_class = min(n_revived, n_passerby)
n_features_est = 40  # 預估特徵數

checks = []
# 整體
checks.append(("整體樣本 >= 10,000", n_usable >= 10000, n_usable))
# 少數類
checks.append(("少數類 >= 1,000", min_class >= 1000, min_class))
# Events per variable (EPV >= 10 for logistic regression)
epv = min_class / n_features_est
checks.append((f"EPV (少數類/{n_features_est}特徵) >= 10", epv >= 10, round(epv, 1)))
# 子群: is_imputed 各組 >= 500
imp_min = imp_group["n"].min()
checks.append(("is_imputed 各子群 >= 500", imp_min >= 500, imp_min))
# 子群: ShopId 各組 >= 100
shop_min = shop_group["n"].min()
checks.append(("ShopId 各子群 >= 100", shop_min >= 100, shop_min))

all_pass = True
for desc, passed, val in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {desc} → 實際值: {val:,}" if isinstance(val, int) else
          f"  [{status}] {desc} → 實際值: {val}")

print()
if all_pass:
    print("  >>> 所有檢查通過，可以往下做 B1~B5 <<<")
else:
    print("  >>> 有檢查未通過，需檢視後決定是否調整 <<<")
print()

# ================================================================
# 11. 輸出 b0_summary.json
# ================================================================
summary = {
    "order_tg_time_range": {
        "min": str(time_range["order_min"].iloc[0]),
        "max": str(time_range["order_max"].iloc[0]),
        "total_rows": int(time_range["total_rows"].iloc[0]),
        "distinct_members": int(time_range["distinct_members"].iloc[0]),
    },
    "t0_candidates": len(t0_df),
    "t0_in_dormant": len(t0_merged),
    "n_usable": n_usable,
    "label": {
        "revived": int(n_revived),
        "passerby": int(n_passerby),
        "base_rate_revived": round(base_rate, 4),
    },
    "by_is_imputed": imp_group.to_dict(orient="records"),
    "by_shop_id": shop_group.to_dict(orient="records"),
    "by_t0_month": month_dist.to_dict(orient="records"),
    "revived_revenue_stats": {
        k: round(float(v), 2) for k, v in rev_stats.items()
    },
    "sample_adequacy": {
        desc: {"pass": passed, "value": val}
        for desc, passed, val in checks
    },
}

out_path = OUT_DIR / "b0_summary.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"已輸出: {out_path}")

con.close()
