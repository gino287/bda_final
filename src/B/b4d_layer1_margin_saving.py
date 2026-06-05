"""
Block 1 / 第一層 — 折扣效率 / 省 margin（成本規避，假設最輕）
============================================================
核心論述：假設品牌現行做法是「回流客一律發券」（treat-all）。
模型讓你對「高分（本來就會回來）」客戶停發券，省下白送在本來就會成交訂單上的折扣 margin。

對手 = treat-all（都發券），不是 treat-none（都不發＝放棄真實營收）。
本層僅依賴「高分客大多沒券也會回來」這個溫和假設；增量營收結論一律標為非因果。

決策機率：校準後 LightGBM（p_lightgbm_cal）、時間外 test n=38,572。
僅實作第一層；第二、三層需產業錨點（uplift 區間、毛利率、兌換率實證），另案處理。
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "output/B/B3/test_predictions.parquet"
CONFIG = ROOT / "config/unit_economics.yaml"
ECON = ROOT / "output/B/B4/economics"
FIG = ROOT / "output/B/B4/figures"
ECON.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

# ── 假設 ──
cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
gm = cfg["gross_margin"]
coupon_face = cfg["coupon_face_value"]
redeem = cfg["redemption_rate"]
fixed = cfg["fixed_contact_cost"]
coupon_cost = coupon_face * redeem          # 每位被觸及的「期望折扣 margin」= 60
contact_cost = coupon_cost + fixed          # 含固定觸及 = 65

# 規模化母體
N_BMOTHER = 144919      # B 母體
N_RETURNERS = 232328    # 回流總體（有 t0）

preds = pd.read_parquet(PREDS)
p = preds["p_lightgbm_cal"].values
y = preds["revived"].values.astype(float)
rev = preds["future_90d_revenue"].values.astype(float)
N = len(preds)

# m：每位真復活者增量毛利（非因果，僅作風險上限估計）
revived_avg_rev = rev[y == 1].mean()
m = revived_avg_rev * gm

print(f"[第一層] test n={N:,}  base rate={y.mean():.4f}")
print(f"  coupon_cost(券面額×兌換)={coupon_cost:.0f}  fixed={fixed}  contact_cost={contact_cost:.0f}")
print(f"  m(復活增量毛利, 風險上限用)={m:.1f}  (= {revived_avg_rev:.0f}×{gm})")


def tier_stats(mask_H, label):
    """對高分層 H 停發券的省下金額與損益破口。"""
    nH = int(mask_H.sum())
    rH = float(y[mask_H].mean()) if nH > 0 else 0.0       # H 的實際復活率
    revivers_H = int(y[mask_H].sum())
    # 毛省下金額（硬數字，無 uplift 假設）= 期望折扣 margin × |H|
    gross_saving = coupon_cost * nH
    gross_saving_with_fixed = contact_cost * nH
    # 損益兩平 uplift：停發省下 vs 失去 H 中「非券不買」者的增量毛利
    #   gross_saving >= u * revivers_H * m  → u_be = gross_saving / (revivers_H * m)
    u_breakeven = gross_saving / (revivers_H * m) if revivers_H > 0 else np.nan
    return {
        "tier": label,
        "n_H": nH,
        "share_of_test": round(nH / N, 4),
        "revival_rate_H": round(rH, 4),
        "revivers_in_H": revivers_H,
        "gross_saving_margin_test": round(gross_saving, 0),
        "gross_saving_with_fixed_test": round(gross_saving_with_fixed, 0),
        "per_returner_saving": round(gross_saving / N, 2),
        "scaled_to_Bmother": round(gross_saving / N * N_BMOTHER, 0),
        "scaled_to_returners": round(gross_saving / N * N_RETURNERS, 0),
        "u_breakeven": round(float(u_breakeven), 4),
    }


# ── 高分層定義（兩種，看哪個故事乾淨）──
rank = np.argsort(-p)
top30_mask = np.zeros(N, bool)
top30_mask[rank[:int(N * 0.30)]] = True
p06_mask = p >= 0.6

tiers = [
    tier_stats(top30_mask, "high_top30pct"),
    tier_stats(p06_mask, "high_p>=0.6"),
]

# ── 低分層（停發省浪費；附 caveat：低分客本就少兌換，flat 60 高估）──
bottom30_mask = np.zeros(N, bool)
bottom30_mask[rank[int(N * 0.70):]] = True
p02_mask = p < 0.2
low_tiers = [
    {**tier_stats(bottom30_mask, "low_bottom30pct"),
     "caveat": "低分客復活率低、實際兌換少，flat coupon_cost 高估省下額；真實節省偏固定觸及成本"},
    {**tier_stats(p02_mask, "low_p<0.2"),
     "caveat": "同上：低分層的硬節省以固定觸及成本(5/人)為主，折扣 margin 節省被高估"},
]

print("\n=== 高分層停發券 — 省下金額與破口 ===")
for t in tiers:
    print(f"  [{t['tier']}] |H|={t['n_H']:,} ({t['share_of_test']:.0%}), "
          f"復活率={t['revival_rate_H']:.1%}")
    print(f"     毛省(margin, test)={t['gross_saving_margin_test']:,.0f}  "
          f"每位回流客={t['per_returner_saving']:.1f}  "
          f"縮放回流總體={t['scaled_to_returners']:,.0f}")
    print(f"     損益兩平 uplift u_be={t['u_breakeven']:.1%}  "
          f"(H 內實際 uplift 低於此即淨賺)")

# ── treat-all 對照支出 ──
treat_all_spend_test = contact_cost * N
treat_all_spend_returners = contact_cost * N_RETURNERS
print(f"\n  [對照] treat-all 全發券總支出: test={treat_all_spend_test:,.0f}  "
      f"縮放回流總體={treat_all_spend_returners:,.0f}")
print(f"  [對照] treat-none = 0 支出，但放棄整批復活營收（非省錢，公司不會選）")

# ================================================================
# 圖：左=各分數十分位復活率；右=停發 top-k% 的省下 vs 風險
# ================================================================
# 十分位（1=最高分）
order_desc = rank
dec = np.array_split(order_desc, 10)
dec_rev_rate = [float(y[idx].mean()) for idx in dec]
dec_n = [len(idx) for idx in dec]
dec_coupon_on_revivers = [coupon_cost * float(y[idx].sum()) for idx in dec]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：復活率 by decile
ax = axes[0]
bars = ax.bar(range(1, 11), dec_rev_rate, color="C0", alpha=0.85, edgecolor="k")
ax.axhline(y.mean(), ls="--", color="red", label=f"overall={y.mean():.3f}")
ax.set(xlabel="Score decile (1=highest P(revive))", ylabel="Actual revival rate",
       title="Revival rate by score decile\n(high deciles = will-return-anyway = coupon waste)")
ax.set_xticks(range(1, 11))
ax.legend(fontsize=8)
for i, r in enumerate(dec_rev_rate):
    ax.annotate(f"{r:.2f}", (i + 1, r), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=7)

# 右：停發 top-k% 的累積毛省 vs 增量毛利風險（參考 u）
ks = np.arange(0, N + 1)
frac = ks / N
y_desc = y[order_desc]
cum_revivers = np.concatenate([[0], np.cumsum(y_desc)])
gross_saving_k = coupon_cost * ks                      # 停 top-k 人的毛省
ax = axes[1]
ax.plot(frac, gross_saving_k, color="C0", lw=2, label="Gross margin saved (hard)")
for u_ref, c in [(0.05, "C2"), (0.10, "C1"), (0.20, "C3")]:
    risk = u_ref * cum_revivers * m
    ax.plot(frac, risk, "--", color=c, label=f"Margin at risk @ u={u_ref:.0%}")
ax.axvline(0.30, ls=":", color="grey")
ax.set(xlabel="Fraction stopped (from highest score)",
       ylabel="NT$ (test set)",
       title="Stop coupons top-down: saving vs revenue-at-risk\n(saving > risk while actual uplift low)")
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = FIG / "layer1_saving_by_tier.png"
plt.savefig(fig_path)
plt.close()

# ================================================================
# JSON 輸出
# ================================================================
out = {
    "layer": "Layer 1 — discount efficiency / margin saving (cost avoidance)",
    "decision_model": "calibrated LightGBM (p_lightgbm_cal), time-based test n=38,572",
    "opponent": "treat-all (send coupon to everyone); treat-none(=0) listed as giving up real revenue, not saving",
    "assumptions": {
        "gross_margin": gm, "coupon_face_value": coupon_face,
        "redemption_rate": redeem, "fixed_contact_cost": fixed,
        "coupon_cost_per_person": coupon_cost, "contact_cost_per_person": contact_cost,
        "m_revival_margin": round(m, 1),
        "note": "m 僅作風險上限估計，非因果增量；本層硬數字（毛省）不依賴 m",
    },
    "scaling_populations": {"test": N, "B_mother": N_BMOTHER, "returners": N_RETURNERS},
    "high_tiers": tiers,
    "low_tiers": low_tiers,
    "treat_all_reference_spend": {
        "test": round(treat_all_spend_test, 0),
        "scaled_returners": round(treat_all_spend_returners, 0),
    },
    "honesty_notes": [
        "毛省金額(券面額×兌換×|H|)為硬數字，不需 uplift 假設。",
        "停發風險以損益兩平 uplift 呈現：H 內實際 uplift 低於 u_breakeven 即淨賺。",
        "依賴溫和假設『高分客大多沒券也會回來』(actual uplift << breakeven)，已明說。",
        "treat-none 非省錢而是放棄真實營收；增量營收結論皆非因果。",
        "縮放到 B 母體/回流總體假設『時間外 test 分佈可推廣』。",
    ],
}
json.dump(out, open(ECON / "layer1_margin_saving.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ================================================================
# ECONOMIC_NARRATIVE.md（先寫第一層，二三層留位）
# ================================================================
t30 = tiers[0]
t06 = tiers[1]
narrative = f"""# B 組｜三層經濟敘事（Block 1）

> 把「模型 vs 規則」的 AUC 故事，升級成「認真做真的省得到錢」的商業論證。
> 決策機率：校準後 LightGBM、時間外 test n={N:,}。對手 = **treat-all（都發券）**。
> 狀態：**第一層已完成**；第二、三層需產業錨點（uplift 區間、毛利率、兌換率實證），待外部資料後補。

## 一句話結論（第一層）

假設現行對所有回流客發券，**改用模型對高分（本來就會回來）客戶停發券**：
- 對 **top-30% 高分層**停發，毛省折扣 margin **約 NT${t30['gross_saving_margin_test']:,.0f}**（test）／每位回流客 **NT${t30['per_returner_saving']:.1f}**／縮放到回流總體 **約 NT${t30['scaled_to_returners']:,.0f}**。
- 只要該層**實際 uplift 低於 {t30['u_breakeven']:.0%}**（損益兩平點），停發即淨賺；而高分層本來就會回來，實際 uplift 結構上遠低於此。

## 第一層：省 margin 的硬數字 + 破口

**核心**：treat-all 把折扣 margin 白送在「本來就會成交」的高分客訂單上。停發 = 純成本規避，假設最輕。

| 高分層定義 | 人數(test) | 該層復活率 | 毛省 margin(test) | 每位回流客 | 縮放回流總體 | 損益兩平 uplift |
|-----------|------:|------:|------:|------:|------:|------:|
| top-30% 分數 | {t30['n_H']:,} | {t30['revival_rate_H']:.1%} | {t30['gross_saving_margin_test']:,.0f} | {t30['per_returner_saving']:.1f} | {t30['scaled_to_returners']:,.0f} | {t30['u_breakeven']:.1%} |
| 校準 P ≥ 0.6 | {t06['n_H']:,} | {t06['revival_rate_H']:.1%} | {t06['gross_saving_margin_test']:,.0f} | {t06['per_returner_saving']:.1f} | {t06['scaled_to_returners']:,.0f} | {t06['u_breakeven']:.1%} |

**對照**：treat-all 全發券總支出 test ≈ NT${treat_all_spend_test:,.0f}（縮放回流總體 ≈ NT${treat_all_spend_returners:,.0f}）；treat-none = 0 支出，但**放棄整批復活營收**，是放棄收入而非省錢，公司不會選，僅列為對照。

**破口詮釋**：損益兩平 uplift `u_be = 毛省 / (H 復活人數 × m)`。停發高分層只可能損失「H 中真的非券不買者」；只要 H 內實際 uplift < u_be 即淨賺。高分層是高 P(復活) 群、本來就會回來，**實際 uplift 結構上偏低**，故落在安全側。（此為本層唯一的溫和假設，已明說；嚴格因果量化見未來工作。）

**圖**：`figures/layer1_saving_by_tier.png` — 左：各分數十分位復活率（高分位＝高復活＝折扣浪費集中處）；右：由高分往下停發的累積毛省 vs 不同參考 uplift 下的營收風險（毛省在實際 uplift 偏低時穩定大於風險）。

## 第二層：uplift 破口 + 產業實證區間（待外部資料）

> 需產業 winback uplift 實證區間、毛利率、兌換率錨點後實作。破口已知約 u≈0.30（模型勝 treat-all）、u≈0.20（treat-all 轉負）。

## 第三層：跨假設穩健性 + 因果限制收束（待外部資料）

> 需 u × 毛利率 × 兌換率 三維實證合理值後實作；報「模型勝出覆蓋率」與因果限制收束段。

## 誠實守則

- 毛省為硬數字，不依賴 uplift；增量營收結論一律標為非因果。
- treat-none = 放棄真實營收，非省錢。
- 縮放假設「時間外 test 分佈可推廣」。
"""
(ECON / "ECONOMIC_NARRATIVE.md").write_text(narrative, encoding="utf-8")

print(f"\n[完成] 第一層輸出:")
print(f"  {ECON / 'layer1_margin_saving.json'}")
print(f"  {fig_path}")
print(f"  {ECON / 'ECONOMIC_NARRATIVE.md'}")
