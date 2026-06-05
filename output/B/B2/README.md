# B2 — 特徵工程

由 `src/B/b2_features.py`（Sprint B2）產生。已含任務 4 的 t0_payment_type 清理。

| 檔案 | 說明 | 狀態 |
|------|------|------|
| `features.parquet` | 144,919 × 54 欄，事件表左接全部特徵；含 leakage 斷言 | ✅ 現行有效（所有建模/分析的根） |

注意：所有 leakage 鐵律（特徵 < t0）在此把關。
