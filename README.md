# Radar XAU/USDT

Bagian dari APK RADAR. Mendeteksi setup intraday emas (OKX `XAU-USDT-SWAP`) dan mengirim alert ke Telegram.

- **Dashboard live:** `index.html` (GitHub Pages). Harga dan candle diambil langsung dari OKX di browser.
- **Alert Telegram:** `scripts/radar_xau.py`, dijalankan GitHub Actions tiap 15 menit saat sesi London/NY dan tiap jam di luar sesi. Mengirim pesan saat setup v1 valid dan saat sinyal kena TP/SL/24 jam.
- **State:** `data/signals.json` (riwayat sinyal = jurnal forward test otomatis), `data/macro.json` (skor makro harian).
- **Evaluasi mingguan:** `scripts/evaluasi_mingguan.py` (workflow *Evaluasi mingguan*, Minggu 08:17 WIB). Backtest ulang aturan v1 di seluruh data OKX terbaru, periode 6/3 bulan & sejak live, cek apakah filter D1/makro masih membantu, sensitivitas SL/TP (hanya dipantau), dan cocokkan sinyal live dengan backtest. Hasil di `data/backtest.json`, dashboard, dan Telegram.
- **Backtest:** `backtest/` (hasil backtest #1, 19 Okt 2026).

## Aturan setup v1 (breakout range Asia)
1. Tren H4: close > EMA50 > EMA200 (long) / kebalikannya (short).
2. Tren D1: EMA20 > EMA50 harian untuk long, < untuk short.
3. Veto makro: skor DXY 5 hari + yield riil 10Y 5 hari (FRED DFII10). Long butuh ≥0, short ≤0.
4. Pemicu: candle H1 yang mulai 14:00–22:00 WIB tutup pertama kali di luar range Asia (07:00–13:59 WIB).
5. Entry open H1 berikutnya, SL 2×ATR14 H1, TP 2R, tutup setelah 24 jam. Satu posisi pada satu waktu.

Backtest Apr 2025–Okt 2026: 82 sinyal, win 55%, +0,26R/sinyal setelah fee. Out-of-sample Mei 2024–Apr 2025: −0,08R. **Status: forward test, belum untuk uang sungguhan.**

## Setup
Secrets repo: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Pages: Settings → Pages → Deploy from branch `main` / root.
Tes manual: tab Actions → Radar XAU → Run workflow.
