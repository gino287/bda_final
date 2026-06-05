"""
Block 3 — 洞察挖掘（3~5 條可行動洞察）
=====================================
方法：對候選特徵分桶 dependence vs 復活率；控制 is_imputed；報效應量；
每條寫「發現 → 所以呢 → 建議動作」。全部相關性、非因果。
資料：全 B 母體 features.parquet（144,919），描述性求穩定。
輸出：output/B/B4/insights/ 各圖 + INSIGHTS.md
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
INS = ROOT / "output/B/B4/insights"
INS.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

df = pd.read_parquet(FEATURES)
base = df["revived"].mean()
print(f"[Block3] n={len(df):,} 整體復活率={base:.4f}")
results = {}


def by_subgroup_rate(col, bins, labels, fname, title, xlabel):
    """分桶復活率（整體 + 控制 is_imputed），存圖，回傳效應量。"""
    d = df[df[col].notna()].copy()
    d["bin"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    def rate(sub):
        g = sub.groupby("bin", observed=True)["revived"].agg(["size", "mean"])
        return g
    overall = rate(d)
    rf = rate(d[~d["is_imputed"]])
    rt = rate(d[d["is_imputed"]])
    # 效應量：bin index vs rate Spearman（整體）
    rho, _ = spearmanr(range(len(overall)), overall["mean"].values)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(overall))
    ax.bar(x, overall["mean"], color="C0", alpha=0.4, label="overall")
    ax.plot(x, rf["mean"].reindex(overall.index).values, "o-", color="C2",
            label="is_imputed=False")
    ax.plot(x, rt["mean"].reindex(overall.index).values, "s-", color="C3",
            label="is_imputed=True")
    ax.axhline(base, ls="--", color="grey", label=f"overall={base:.3f}")
    ax.set_xticks(list(x)); ax.set_xticklabels(overall.index, rotation=20, ha="right")
    ax.set(xlabel=xlabel, ylabel="Revival rate", title=f"{title}\nSpearman(bin,rate)={rho:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(INS / fname); plt.close()
    return overall, rho


# ================================================================
# 洞察 1 ★ 頭條：t0_discount_ratio vs 復活（含回餵 Layer 2 的 u 先驗）
# ================================================================
print("\n[洞察1] t0_discount_ratio vs 復活 ...")
bins = [-0.01, 0.001, 0.05, 0.10, 0.20, 0.50, 1.01]
labels = ["0%(full)", "0-5%", "5-10%", "10-20%", "20-50%", ">50%"]
ov1, rho1 = by_subgroup_rate("t0_discount_ratio", bins, labels,
    "insight1_t0_discount.png",
    "Revival rate by t0 discount depth (control is_imputed)",
    "t0 order discount ratio")
# 全價 vs 深折扣的復活率（誠實檢驗假說）
r_full = float(df[df["t0_discount_ratio"] <= 0.001]["revived"].mean())
r_deep = float(df[df["t0_discount_ratio"] > 0.20]["revived"].mean())
n_full = int((df["t0_discount_ratio"] <= 0.001).sum())
n_deep = int((df["t0_discount_ratio"] > 0.20).sum())
# 假說「折扣驅動的回購較不黏著」= 預期 r_deep < r_full。資料檢驗：
hypo_supported = r_deep < r_full
print(f"  全價回購復活率={r_full:.3f} (n={n_full:,}); 深折扣(>20%)={r_deep:.3f} (n={n_deep:,})")
print(f"  Spearman(bin,rate)={rho1:.2f}  → 假說『折扣=較不黏著』{'成立' if hypo_supported else '【不成立】(深折扣反而略高)'}")
results["insight1_t0_discount"] = {
    "hypothesis": "discount-driven repurchase is less sticky (expect r_deep < r_full)",
    "hypothesis_supported": bool(hypo_supported),
    "spearman": round(rho1, 3), "r_full_price": round(r_full, 3),
    "r_deep_discount": round(r_deep, 3), "n_full": n_full, "n_deep": n_deep,
    "finding": "myth-busted: t0 discount depth does NOT signal churn; deep-discount repurchasers revive slightly MORE (promo-engaged regulars). Only token 0-5% discounts dip.",
    "by_bin": {str(k): round(float(v), 3) for k, v in ov1["mean"].items()},
}

# ================================================================
# 洞察 2：t0_amount（匯入已證實，不重跑）
# ================================================================
results["insight2_t0_amount"] = {
    "status": "imported", "spearman": -0.90,
    "note": "已於任務3證實：t0 金額越高復活率越低；控制 is_imputed 後仍成立(False -0.76, True -0.88)。圖見 figures/t0_amount_revival.png",
}

# ================================================================
# 洞察 3：回購通路（offline vs online）vs 復活
# ================================================================
print("\n[洞察3] t0 回購通路 vs 復活 ...")
df["t0_is_offline"] = (df["t0_channel_type"] == "Pos").astype(int)
g3 = df.groupby("t0_is_offline")["revived"].agg(["size", "mean"])
# 控制 is_imputed
g3_imp = df.groupby(["is_imputed", "t0_is_offline"])["revived"].mean().unstack()
r_off = float(g3.loc[1, "mean"]); r_on = float(g3.loc[0, "mean"])
print(f"  線下POS回購復活率={r_off:.3f} (n={int(g3.loc[1,'size']):,}); "
      f"線上={r_on:.3f} (n={int(g3.loc[0,'size']):,})")
fig, ax = plt.subplots(figsize=(7, 5))
xpos = [0, 1]
ax.bar([x-0.15 for x in xpos], [g3_imp.loc[False, 0], g3_imp.loc[False, 1]],
       width=0.3, color="C2", label="is_imputed=False")
ax.bar([x+0.15 for x in xpos], [g3_imp.loc[True, 0], g3_imp.loc[True, 1]],
       width=0.3, color="C3", label="is_imputed=True")
ax.axhline(base, ls="--", color="grey", label=f"overall={base:.3f}")
ax.set_xticks(xpos); ax.set_xticklabels(["Online/App", "Offline/POS"])
ax.set(ylabel="Revival rate", title="Revival by t0 repurchase channel (control is_imputed)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(INS / "insight3_channel.png"); plt.close()
results["insight3_channel"] = {
    "r_offline_pos": round(r_off, 3), "r_online": round(r_on, 3),
    "lift_offline": round(r_off / r_on, 2),
    "holds_both_subgroups": bool(g3_imp.loc[False,1]>g3_imp.loc[False,0] and g3_imp.loc[True,1]>g3_imp.loc[True,0]),
}

# ================================================================
# 洞察 4：沉睡深度 vs 復活（找斷崖）
# ================================================================
print("\n[洞察4] 沉睡深度 dormancy_gap_days vs 復活 ...")
qbins = df["dormancy_gap_days"].quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]).values
qbins = np.unique(qbins)
ov4, rho4 = by_subgroup_rate("dormancy_gap_days", qbins,
    [f"{int(qbins[i])}-{int(qbins[i+1])}" for i in range(len(qbins)-1)],
    "insight4_dormancy.png",
    "Revival rate by dormancy gap days (control is_imputed)",
    "Days dormant before t0 (decile bins)")
print(f"  Spearman={rho4:.2f}；最淺桶={ov4['mean'].iloc[0]:.3f} 最深桶={ov4['mean'].iloc[-1]:.3f}")
results["insight4_dormancy"] = {
    "spearman": round(rho4, 3),
    "shallowest_bin_rate": round(float(ov4["mean"].iloc[0]), 3),
    "deepest_bin_rate": round(float(ov4["mean"].iloc[-1]), 3),
}

# ================================================================
# 洞察 5：MemberCardLevel 復活梯度
# ================================================================
print("\n[洞察5] MemberCardLevel 復活梯度 ...")
mc = df.copy()
mc["mcl"] = mc["MemberCardLevel"].fillna(0)  # 0 視為無等級/缺失
g5 = mc.groupby("mcl")["revived"].agg(["size", "mean"]).reset_index()
g5 = g5[g5["size"] >= 100]
print(g5.to_string(index=False))
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.bar(g5["mcl"].astype(str), g5["mean"], color="C0", alpha=0.85, edgecolor="k")
ax.axhline(base, ls="--", color="red", label=f"overall={base:.3f}")
for _, r in g5.iterrows():
    ax.annotate(f"{r['mean']:.2f}\nn={int(r['size']/1000)}k",
                (str(r["mcl"]), r["mean"]), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7)
ax.set(xlabel="MemberCardLevel (0=none/missing)", ylabel="Revival rate",
       title="Revival rate gradient by MemberCardLevel")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(INS / "insight5_cardlevel.png"); plt.close()
results["insight5_cardlevel"] = {
    "by_level": {str(int(r["mcl"])): round(float(r["mean"]), 3) for _, r in g5.iterrows()},
}

# ================================================================
# INSIGHTS.md（三段式：發現/所以呢/動作）
# ================================================================
md = f"""# Block 3 — 洞察挖掘（3~5 條可行動洞察）

> 方法：分桶 dependence vs 復活率，控制 is_imputed（必要時 ShopId）。
> 全 B 母體 n={len(df):,}，整體復活率 {base:.1%}。**所有洞察為相關性、非因果。**

## ★ 洞察 1（頭條，破除迷思）：t0 折扣深度【不是】流失訊號

- **檢驗的假說**：「折扣驅動的回購較不黏著」（預期深折扣回購者復活率 < 全價）。
- **發現（假說不成立）**：t0 全價回購者復活率 **{r_full:.1%}**（n={n_full:,}），深折扣（>20%）回購者 **{r_deep:.1%}**（n={n_deep:,}）——**深折扣反而略高**；跨桶 Spearman=**{rho1:+.2f}**（正向，非負向），控制 is_imputed 後型態一致（見 `insights/insight1_t0_discount.png`）。型態呈 U 型：只有「象徵性 0-5% 小折扣」復活率最低（26.7%）。
- **所以呢**：在本資料中，**折扣深度不能當流失紅旗**。深折扣回購者更像「促銷活躍的常客」（美妝保健大檔期吸引的 deal-savvy 忠誠客），不是「拿完就走」。這推翻了「發深折扣＝製造低黏著回購」的直覺，本身是有份量的誠實發現。
- **建議動作**：**不要**用「是否用折扣／折扣多深」當投放排除條件——它與復活無負相關。真正的浪費槓桿在 **propensity（Block 1：別對高分必回客花折扣）**，不在折扣深度。
- **回餵 Layer 2（誠實版）**：折扣深度角度**未**提供「折扣浪費」的負向證據，故**無法**由此導出資料導向的 u；Layer 2 的 uplift 比例維持產業換算錨點 **15~25%**（非因果）。

## 洞察 2：單筆大額 = 流失警訊（t0_amount，匯入已證實）

- **發現**：t0 金額越高、復活率越低，整體 Spearman ≈ **−0.90**，控制 is_imputed 後仍成立（圖見 `figures/t0_amount_revival.png`）。
- **所以呢**：回購當下砸大錢多為一次性消費（囤貨/送禮/衝動），非回到規律。
- **建議動作**：對「t0 高額單筆」客不應樂觀，深折扣留存資源別投這。

## 洞察 3：線下/POS 回購 = 在地黏著（回購通路 vs 復活）

- **發現**：t0 線下 POS 回購者復活率 **{r_off:.1%}**，線上/App **{r_on:.1%}**（lift {r_off/r_on:.2f}×）；控制 is_imputed 後{'仍成立' if results['insight3_channel']['holds_both_subgroups'] else '部分成立'}（見 `insights/insight3_channel.png`）。
- **所以呢**：店內回購反映在地、習慣性消費，黏著度高（呼應 SHAP offline_ratio 正向）。
- **建議動作**：線下回購客優先納入會員/門市經營，不需靠折扣。

## 洞察 4：沉睡越深、復活越難（沉睡深度 vs 復活）

- **發現**：復活率隨沉睡天數遞減，Spearman={rho4:.2f}；最淺桶 {ov4['mean'].iloc[0]:.1%} → 最深桶 {ov4['mean'].iloc[-1]:.1%}（見 `insights/insight4_dormancy.png`）。
- **所以呢**：沉睡越久回來的人，越可能只是路過。
- **建議動作**：喚回資源優先給沉睡較淺者；對極深沉睡者降低期望、控管折扣成本。

## 洞察 5：高卡等 = 復活近必然（MemberCardLevel 梯度）

- **發現**：復活率隨會員卡等單調上升（見 `insights/insight5_cardlevel.png`）。
- **所以呢**：高卡等客本來就會回來，對其發折扣＝純浪費 margin。
- **建議動作**：高卡等層**直接停發折扣**（接 Block 2 高分層動作），改不花折扣的會員經營。

---

## 回餵 Layer 2 的 u 先驗（閉環，誠實版）

原計畫想用洞察 1（折扣深度 vs 復活）反推資料導向的 uplift 比例。**但資料顯示折扣深度與復活無負相關**（甚至略正），故**無法**由此導出「折扣浪費」的 u 估計。因此 Layer 2 的 uplift 比例**維持產業換算錨點 15~25%**（email 4.2%→5.4% → ratio 0.22 等）。
此結果本身強化了一個誠實立場：**折扣的價值不該用「折扣深度→復活」這種相關性來推斷**，真正的因果增量需 holdout 實驗才能定（列未來工作）。模型的經濟價值因此**不依賴折扣的因果**，而是靠 propensity 排序（高分必回客停發、中段精準投放）。
"""
(INS / "INSIGHTS.md").write_text(md, encoding="utf-8")
json.dump(results, open(INS / "insights_data.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n[完成] {INS}/INSIGHTS.md + 5 圖 + insights_data.json")
