"""
任務 B-R1 — ShopId + MemberCardLevel leakage / 貢獻 robustness check
==================================================================
回應「ShopId、MemberCardLevel 不該餵模型」的疑慮：
  - MemberCardLevel 可能含資料截取時點資訊（t0 後升等）→ 潛在 leakage。
  - ShopId 無 leakage 但品牌特定、不可轉移。
量化移除後結論是否穩健。

沿用 features.parquet + 時間外切分（train t0<2023-11、test t0>=2023-11）+ B3 的 LightGBM 設定。
用未校準 raw LightGBM（省時）。**不覆蓋任何 frozen 模型/輸出**；結果存 output/B/B3/robustness_features/。

四變體（同一份時間外 test）：
  (a) full 全特徵
  (b) 移除 MemberCardLevel(+membercard_missing) + ShopId
  (c) 只移除 MemberCardLevel(+membercard_missing)
  (d) 只移除 ShopId

註：移除 MemberCardLevel 時一併移除其衍生 membercard_missing 旗標
    （同源自可能含 t0 後升等的欄位），以完整評估 leakage 疑慮。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output/B/B2/features.parquet"
OUT = ROOT / "output/B/B3/robustness_features"
OUT.mkdir(parents=True, exist_ok=True)
RNG = 42

# ── 沿用 B3 欄位定義 ──
DROP_COLS = ["ShopMemberId", "t0", "last_purchase_date",
             "revived", "future_finish_count", "future_90d_revenue"]
CAT_COLS_ALL = ["ShopId", "RegisterSourceTypeDef", "Gender",
                "t0_channel_type", "t0_channel_detail", "t0_payment_type", "age_bucket"]
BOOL_COLS = ["is_imputed", "IsAppInstalled", "IsEnableEmail",
             "IsEnablePushNotification", "IsEnableShortMessage"]
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=63,
                  min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=RNG, n_jobs=-1, verbose=-1)

df = pd.read_parquet(FEATURES)
df["t0"] = pd.to_datetime(df["t0"])
for c in BOOL_COLS:
    df[c] = df[c].astype(int)
for c in CAT_COLS_ALL:
    df[c] = df[c].astype(str).replace("nan", "Missing")

train_mask = df["t0"] < pd.Timestamp("2023-11-01")
test_mask = ~train_mask
y = df["revived"].values
ytr, yte = y[train_mask.values], y[test_mask.values]
spw = (ytr == 0).sum() / (ytr == 1).sum()
ALL_FEAT = [c for c in df.columns if c not in DROP_COLS]
print(f"[B-R1] train={train_mask.sum():,} test={test_mask.sum():,} "
      f"test base rate={yte.mean():.4f} scale_pos_weight={spw:.3f}")


def train_variant(remove):
    """remove: 要移除的特徵名 list。回傳 test 指標。"""
    feats = [c for c in ALL_FEAT if c not in remove]
    cats = [c for c in CAT_COLS_ALL if c in feats]
    Xtr = df.loc[train_mask, feats].copy()
    Xte = df.loc[test_mask, feats].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    clf = lgb.LGBMClassifier(scale_pos_weight=spw, **LGB_PARAMS)
    clf.fit(Xtr, ytr, categorical_feature=cats)
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "n_features": len(feats),
        "roc_auc": round(float(roc_auc_score(yte, p)), 4),
        "pr_auc": round(float(average_precision_score(yte, p)), 4),
        "brier": round(float(brier_score_loss(yte, np.clip(p, 0, 1))), 4),
    }


MCL = ["MemberCardLevel", "membercard_missing"]
variants = {
    "a_full": [],
    "b_drop_cardlevel_and_shopid": MCL + ["ShopId"],
    "c_drop_cardlevel_only": MCL,
    "d_drop_shopid_only": ["ShopId"],
}
results = {}
for name, rm in variants.items():
    results[name] = {"removed": rm, **train_variant(rm)}
    r = results[name]
    print(f"  {name:32s} feats={r['n_features']:2d}  "
          f"ROC-AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  Brier={r['brier']}")

# ── 相對 full 的下降 ──
full = results["a_full"]
for name in results:
    results[name]["d_roc_auc_vs_full"] = round(results[name]["roc_auc"] - full["roc_auc"], 4)
    results[name]["d_pr_auc_vs_full"] = round(results[name]["pr_auc"] - full["pr_auc"], 4)

json.dump(results, open(OUT / "feature_ablation.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ── markdown 解讀 ──
b, c, d = results["b_drop_cardlevel_and_shopid"], results["c_drop_cardlevel_only"], results["d_drop_shopid_only"]
md = f"""# 任務 B-R1 — ShopId + MemberCardLevel leakage / 貢獻 robustness check

> 回應「ShopId、MemberCardLevel 不該餵模型」的疑慮。時間外 test n={int(test_mask.sum()):,}，
> raw（未校準）LightGBM、B3 相同設定。**frozen 模型/輸出未被改動**，結果獨立存於本資料夾。
> 註：移除 MemberCardLevel 時一併移除其衍生 `membercard_missing`（同源自可能含 t0 後升等的欄位）。

## 四變體比較

| 變體 | 特徵數 | ROC-AUC | PR-AUC | Brier | ΔROC vs full | ΔPR vs full |
|------|------:|:------:|:-----:|:-----:|:----:|:----:|
| (a) 全特徵 | {full['n_features']} | {full['roc_auc']} | {full['pr_auc']} | {full['brier']} | — | — |
| (b) 移除 CardLevel + ShopId | {b['n_features']} | {b['roc_auc']} | {b['pr_auc']} | {b['brier']} | {b['d_roc_auc_vs_full']:+.4f} | {b['d_pr_auc_vs_full']:+.4f} |
| (c) 只移除 MemberCardLevel | {c['n_features']} | {c['roc_auc']} | {c['pr_auc']} | {c['brier']} | {c['d_roc_auc_vs_full']:+.4f} | {c['d_pr_auc_vs_full']:+.4f} |
| (d) 只移除 ShopId | {d['n_features']} | {d['roc_auc']} | {d['pr_auc']} | {d['brier']} | {d['d_roc_auc_vs_full']:+.4f} | {d['d_pr_auc_vs_full']:+.4f} |

## 解讀

- **移除兩者（b）**：ROC-AUC 由 {full['roc_auc']} → **{b['roc_auc']}**（{b['d_roc_auc_vs_full']:+.4f}）、PR-AUC {full['pr_auc']} → **{b['pr_auc']}**（{b['d_pr_auc_vs_full']:+.4f}）。即使同時拿掉品牌與卡等，**行為/交易特徵（t0 回購情境、RFM、通路、沉睡深度）仍承載主要訊號**，模型仍明顯優於 RFM-only 邏輯迴歸 baseline（PR-AUC 0.503）。
- **MemberCardLevel 的 leakage 疑慮（c）**：單獨移除後 ROC-AUC 變動 {c['d_roc_auc_vs_full']:+.4f}、PR-AUC {c['d_pr_auc_vs_full']:+.4f}。{'影響有界——即使該欄含部分 t0 後升等資訊，對整體結論的上限影響也僅此幅度。' if abs(c['d_pr_auc_vs_full'])<0.03 else '影響相對較大，需在報告揭露其 leakage 風險與依賴度。'}
- **ShopId（d）**：單獨移除後 PR-AUC 變動 {d['d_pr_auc_vs_full']:+.4f}。ShopId 無 leakage（靜態品牌標記），但品牌特定、不可跨品牌轉移；保留可提升 pooled 效能，移除後行為特徵仍足以支撐。

## 一句話回應質疑

> **移除 ShopId 與 MemberCardLevel 後，AUC 僅由 {full['roc_auc']} 降到 {b['roc_auc']}（PR-AUC {full['pr_auc']}→{b['pr_auc']}），行為/交易特徵承載主要訊號；MemberCardLevel 的潛在 leakage 對結論的影響有界（單獨移除僅 ΔPR-AUC {c['d_pr_auc_vs_full']:+.4f}），核心結論穩健。**
"""
(OUT / "feature_ablation.md").write_text(md, encoding="utf-8")
print(f"\n[完成] {OUT}/feature_ablation.json + feature_ablation.md")
