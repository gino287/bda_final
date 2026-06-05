"""
Block 2 — 客戶分級 + 策略地圖 + 品牌差異化部署
==============================================
把分數變成可部署的分群與行動方案，串接 Block 1 政策與 Block 3 洞察。
決策機率：校準後 LightGBM、時間外 test n=38,572。

2A 三層分級（高/中/低，沿用 Block 1 門檻）
2B 二維策略地圖（propensity × 歷史價值，3×3）
2C 品牌差異化部署（用 brand_heterogeneity，含小樣本警示）
輸出：output/B/B5/segmentation/
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
FEATURES = ROOT / "output/B/B2/features.parquet"
BRAND = ROOT / "output/B/B5/report_assets/brand_heterogeneity.json"
SEG = ROOT / "output/B/B5/segmentation"
SEG.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

preds = pd.read_parquet(PREDS)
feat = pd.read_parquet(FEATURES, columns=["ShopMemberId", "ShopId",
        "hist_total_amount", "any_marketing_reachable"])
d = preds.merge(feat, on=["ShopMemberId", "ShopId"], how="left")
p = d["p_lightgbm_cal"].values
N = len(d)
print(f"[Block2] n={N:,} 整體復活率={d['revived'].mean():.4f}")

# ================================================================
# 2A. 三層分級（score 三分位：高 top30% / 中 40% / 低 30%）
# ================================================================
hi_thr = np.quantile(p, 0.70)
lo_thr = np.quantile(p, 0.30)
d["tier"] = np.where(p >= hi_thr, "High", np.where(p < lo_thr, "Low", "Mid"))

tier_rows = []
for t in ["High", "Mid", "Low"]:
    s = d[d["tier"] == t]
    tier_rows.append({
        "tier": t, "n": len(s), "share": round(len(s)/N, 3),
        "revival_rate": round(s["revived"].mean(), 4),
        "avg_future_value": round(s["future_90d_revenue"].mean(), 1),
        "avg_future_value_revived": round(s[s["revived"]==1]["future_90d_revenue"].mean(), 1),
        "reachable_share": round(s["any_marketing_reachable"].fillna(0).mean(), 3),
        "p_range": f"{s['p_lightgbm_cal'].min():.2f}-{s['p_lightgbm_cal'].max():.2f}",
    })
actions = {
    "High": "停發折扣，改不花折扣的留存（會員升級/跨售）→ 省 margin",
    "Mid": "精準投放留存資源（券發這裡 ROI 最高，profit-max τ* 落點）",
    "Low": "預設放生省觸及成本；高價值低分者改走『搶救』（見策略地圖）",
}
tier_df = pd.DataFrame(tier_rows)
tier_df["action"] = tier_df["tier"].map(actions)
tier_df.to_csv(SEG / "tier_table.csv", index=False, encoding="utf-8-sig")
print("\n=== 2A 三層分級 ===")
print(tier_df.to_string(index=False))

# ================================================================
# 2B. 二維策略地圖：propensity(P) × 歷史價值(hist_total_amount) 3×3
# ================================================================
d["val"] = d["hist_total_amount"].fillna(0)
p_terc = np.quantile(p, [1/3, 2/3])
v_terc = np.quantile(d["val"], [1/3, 2/3])
def terc(x, q): return np.where(x < q[0], 0, np.where(x < q[1], 1, 2))
d["p_t"] = terc(p, p_terc)
d["v_t"] = terc(d["val"].values, v_terc)

# 每格：人數、復活率、平均未來價值
cell_n = np.zeros((3, 3)); cell_rev = np.zeros((3, 3)); cell_val = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        s = d[(d["v_t"] == i) & (d["p_t"] == j)]
        cell_n[i, j] = len(s)
        cell_rev[i, j] = s["revived"].mean() if len(s) else 0
        cell_val[i, j] = s["future_90d_revenue"].mean() if len(s) else 0

# 象限命名（英文避免字型缺字）
names = [
    ["LowV-LowP\nlet go", "LowV-MidP\nlow priority", "LowV-HighP\nself-return (no coupon)"],
    ["MidV-LowP\nwatch", "MidV-MidP\n* precision target", "MidV-HighP\nretain no-discount"],
    ["HiV-LowP\n** HIGH-VALUE CHURN\n(retention priority)", "HiV-MidP\nkey precision target", "HiV-HighP\nVIP protect (no discount)"],
]
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cell_rev, cmap="RdYlGn", origin="lower", aspect="auto", vmin=0, vmax=0.7)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{names[i][j]}\nn={int(cell_n[i,j]):,}\nrev={cell_rev[i,j]:.0%}",
                ha="center", va="center", fontsize=8)
ax.set_xticks([0,1,2]); ax.set_xticklabels(["Low P", "Mid P", "High P"])
ax.set_yticks([0,1,2]); ax.set_yticklabels(["Low value", "Mid value", "High value"])
ax.set_xlabel("Revival propensity (calibrated P)")
ax.set_ylabel("Historical value (hist_total_amount)")
ax.set_title("Block 2: Strategy Map — propensity × value (cell color = revival rate)")
fig.colorbar(im, ax=ax, label="Revival rate")
plt.tight_layout(); plt.savefig(SEG / "strategy_map.png"); plt.close()

# 高價值流失風險格（高值低P）數據
hv_lp = d[(d["v_t"] == 2) & (d["p_t"] == 0)]
print(f"\n=== 2B 高價值流失風險（高值×低P）===")
print(f"  n={len(hv_lp):,} 復活率={hv_lp['revived'].mean():.3f} "
      f"平均歷史額={hv_lp['hist_total_amount'].mean():.0f} → 留存預算最該花")

# ================================================================
# 2C. 品牌差異化部署
# ================================================================
brand = json.load(open(BRAND, encoding="utf-8"))
bt = pd.DataFrame(brand["table"])
brand_md = f"""# Block 2C — 品牌差異化部署

> 用 pooled 模型 + 用品牌決定「在哪部署」，**不是每品牌各訓一個**。
> 資料：`B5/report_assets/brand_heterogeneity.*`（時間外 test）。

| 品牌 | n | 復活率 | ROC-AUC | 第1分位 lift | 部署建議 |
|------|--:|------:|:------:|:----:|------|
"""
for _, r in bt.sort_values("base_rate").iterrows():
    if r["base_rate"] < 0.25:
        rec = "優先導入模型精準投放（lift 高、ROI 最高）"
    elif r["base_rate"] > 0.45:
        rec = "客人本來就會回，別過度投資 targeting；重點改成別過度打折"
    else:
        rec = "標準部署：模型分群 + 中段投放"
    warn = " ⚠️小樣本不可靠" if r["n"] < 3000 or r["roc_auc"] < 0.62 else ""
    brand_md += (f"| {r['ShopId_short']} | {int(r['n']):,} | {r['base_rate']:.1%} | "
                 f"{r['roc_auc']:.2f} | {r['decile1_lift']:.2f}× | {rec}{warn} |\n")
brand_md += f"""
**誠實警示**：小樣本品牌（如 NOmceSRC n≈2.4K、ROC-AUC≈0.59）per-brand 指標不可靠；
結論是「**pooled 模型 + 用品牌決定部署優先序**」，不是每品牌各訓一個模型。

**部署優先序**：
1. **低復活率品牌（RZSHERLB 12.9%, lift 4.14×）**：模型鎖定價值最大 → 優先導入精準投放。
2. **中復活率品牌（hFwniXiB, zXQPxhiL）**：標準模型分群部署。
3. **高復活率品牌（3WUOySTy 48.5%）**：客人本來就會回 → 別過度 targeting，重點放在「別過度打折」省 margin。
"""
(SEG / "brand_deployment.md").write_text(brand_md, encoding="utf-8")

# ================================================================
# PLAYBOOK.md
# ================================================================
hi = tier_df[tier_df["tier"]=="High"].iloc[0]
mid = tier_df[tier_df["tier"]=="Mid"].iloc[0]
low = tier_df[tier_df["tier"]=="Low"].iloc[0]
playbook = f"""# Block 2 — 行動方案 PLAYBOOK

> 校準後 LightGBM、時間外 test n={N:,}。對齊 Block 1 政策（高分停發/中段投放/低分放生）。

## 三層分級行動表

| 層 | 人數(占比) | 復活率 | 平均未來價值 | 可觸及率 | 動作 |
|----|------:|------:|------:|------:|------|
| **高分** | {int(hi['n']):,} ({hi['share']:.0%}) | {hi['revival_rate']:.1%} | {hi['avg_future_value']:.0f} | {hi['reachable_share']:.0%} | {actions['High']} |
| **中分** | {int(mid['n']):,} ({mid['share']:.0%}) | {mid['revival_rate']:.1%} | {mid['avg_future_value']:.0f} | {mid['reachable_share']:.0%} | {actions['Mid']} |
| **低分** | {int(low['n']):,} ({low['share']:.0%}) | {low['revival_rate']:.1%} | {low['avg_future_value']:.0f} | {low['reachable_share']:.0%} | {actions['Low']} |

## 策略地圖九宮格重點（propensity × 歷史價值）

- **★★ 高值 × 低P = 高價值流失風險**（n={len(hv_lp):,}，復活率 {hv_lp['revived'].mean():.0%}）：**留存預算最該花的地方**——大戶但模型判斷不會自己回來，值得下重本搶救（一對一、專屬優惠、專人聯繫）。
- **VIP：高值 × 高P**：保護、**別打折**（本來就會回，打折純浪費 margin）。
- **★ 中值 × 中P = 精準投放主場**：券發這裡 ROI 最高（profit-max τ* 落點）。
- **低值 × 低P**：放生，省觸及成本。

## 與洞察（Block 3）的連結
- 高分層「停發折扣」呼應洞察 5（高卡等復活近必然）。
- 中段投放對象避開「t0 單筆大額」（洞察 2，流失警訊）與「極深沉睡」（洞察 4）。
- 線下/POS 回購者（洞察 3，黏著）多落高分層 → 走會員經營而非折扣。
- 折扣深度**不**作為排除條件（洞察 1 破除迷思：折扣深度非流失訊號）。

## 品牌部署
見 `brand_deployment.md`：低復活率品牌優先導入模型；高復活率品牌重點改為「別過度打折」。
"""
(SEG / "PLAYBOOK.md").write_text(playbook, encoding="utf-8")

json.dump({
    "tiers": tier_rows,
    "high_value_low_p": {"n": len(hv_lp), "revival_rate": round(float(hv_lp["revived"].mean()),3),
                         "avg_hist_amount": round(float(hv_lp["hist_total_amount"].mean()),0)},
    "strategy_map_revival": cell_rev.round(3).tolist(),
    "strategy_map_n": cell_n.astype(int).tolist(),
}, open(SEG / "segmentation_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\n[完成] {SEG}/: tier_table.csv, strategy_map.png, brand_deployment.md, PLAYBOOK.md")
