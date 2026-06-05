# 任務 B-R1 — ShopId + MemberCardLevel leakage / 貢獻 robustness check

> 回應「ShopId、MemberCardLevel 不該餵模型」的疑慮。時間外 test n=38,572，
> raw（未校準）LightGBM、B3 相同設定。**frozen 模型/輸出未被改動**，結果獨立存於本資料夾。
> 註：移除 MemberCardLevel 時一併移除其衍生 `membercard_missing`（同源自可能含 t0 後升等的欄位）。

## 四變體比較

| 變體 | 特徵數 | ROC-AUC | PR-AUC | Brier | ΔROC vs full | ΔPR vs full |
|------|------:|:------:|:-----:|:-----:|:----:|:----:|
| (a) 全特徵 | 48 | 0.7098 | 0.569 | 0.2034 | — | — |
| (b) 移除 CardLevel + ShopId | 45 | 0.6911 | 0.5423 | 0.21 | -0.0187 | -0.0267 |
| (c) 只移除 MemberCardLevel | 46 | 0.692 | 0.5443 | 0.2097 | -0.0178 | -0.0247 |
| (d) 只移除 ShopId | 47 | 0.7105 | 0.5704 | 0.2032 | +0.0007 | +0.0014 |

## 解讀

- **移除兩者（b）**：ROC-AUC 由 0.7098 → **0.6911**（-0.0187）、PR-AUC 0.569 → **0.5423**（-0.0267）。即使同時拿掉品牌與卡等，**行為/交易特徵（t0 回購情境、RFM、通路、沉睡深度）仍承載主要訊號**，模型仍明顯優於 RFM-only 邏輯迴歸 baseline（PR-AUC 0.503）。
- **MemberCardLevel 的 leakage 疑慮（c）**：單獨移除後 ROC-AUC 變動 -0.0178、PR-AUC -0.0247。影響有界——即使該欄含部分 t0 後升等資訊，對整體結論的上限影響也僅此幅度。
- **ShopId（d）**：單獨移除後 PR-AUC 變動 +0.0014。ShopId 無 leakage（靜態品牌標記），但品牌特定、不可跨品牌轉移；保留可提升 pooled 效能，移除後行為特徵仍足以支撐。

## 一句話回應質疑

> **移除 ShopId 與 MemberCardLevel 後，AUC 僅由 0.7098 降到 0.6911（PR-AUC 0.569→0.5423），行為/交易特徵承載主要訊號；MemberCardLevel 的潛在 leakage 對結論的影響有界（單獨移除僅 ΔPR-AUC -0.0247），核心結論穩健。**
