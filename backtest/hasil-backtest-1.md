# Radar XAU/USDT — Kerangka & Backtest #1 (19 Okt 2026)

Radar keempat APK RADAR, terpisah dari Radar BTC/IDX/US. Artifact: https://claude.ai/artifact/VGYq3rcHm5bpJuKvJRugf9 (db: koleksi `narasi` = catatan narasi per tanggal, `jurnal` = paper trade).

## Tujuan (dari Indra, 19 Okt 2026)
Trading di crypto exchange, tahu arah market dan ambil posisi lebih efektif, target XAU/USDT seperti di MetaTrader, gabungan narasi dunia + teknikal. Pilihan: intraday (M15–H1), prototipe artifact dulu, skor makro otomatis + catatan manual.

## Data
- OKX `XAU-USDT-SWAP` (listing 9 Apr 2025, ctVal 0.001 XAU). Candle via `www.okx.com/api/v5/market/history-candles` (bisa mundur sampai listing, 15m). Funding ~0, tidak berguna. CORS OKX mengizinkan origin claude.ai/browser.
- Makro: Yahoo (DX-Y.NYB, ^TNX, ^VIX, GLD, GC=F) dan FRED `DFII10` (yield riil 10Y, fredgraph.csv jalan via curl).
- Out-of-sample: Yahoo GC=F 1h (730 hari, mulai Mei 2024).
- `fapi.binance.com` & Bybit geo-block dari container.
- **Batasan artifact:** CSP memblokir fetch eksternal, jadi halaman tidak bisa live. Versi live harus single-file HTML di GitHub Pages, atau update via scheduled task.

## Aturan v1 — breakout range Asia
1. Tren H4: long jika close > EMA50 > EMA200 (short kebalikannya).
2. Tren D1: EMA20 > EMA50 harian (candle kemarin) untuk long.
3. Veto makro: skor = DXY 5h (≤−0,5% → +1, ≥+0,5% → −1) + DFII10 5h (≤−0,05 → +1, ≥+0,05 → −1); long butuh ≥0, short ≤0. Data makro diberi lag 2 hari di backtest.
4. Pemicu: H1 tutup pertama kali di luar range Asia (00–07 UTC = 07:00–13:59 WIB), candle mulai 07–15 UTC (14:00–22:00 WIB).
5. Entry open H1 berikutnya, SL 2×ATR14 H1, TP 2R, time stop 24 jam, satu posisi per waktu, tidak entry Jumat ≥16 UTC & akhir pekan.

## Hasil (SL 2ATR / TP 2R / 24j, fee taker 0,1% PP)
- OKX Apr 2025–Okt 2026: v1 n=82, win 54,9%, +0,26R/trade net (gross +0,43, maker +0,36), PF 1,53, maxDD −6,6R; 2025 +0,155, 2026 +0,361; 6 dari 7 kuartal positif (Q3 2026 −4,7R). ~4,5 trade/bulan, median hold 14 jam.
- Pembanding: masuk tiap jam searah tren H4 gross +0,14 / net −0,05. Pullback EMA, Fibonacci, RSI reset tidak lebih baik dari tren; kontra-tren RSI −0,44R. Breakout Asia tanpa filter tren −0,13R.
- **Out-of-sample GC=F Mei 2024–Apr 2025: v1 n=60, gross +0,12R, net −0,083R (PF 0,87). Gagal setelah fee.** Short OOS 8 trade semua rugi.
- Paling konsisten di dua periode: veto makro (breakout melawan makro −0,29R n=34 in-sample; −0,13R gross n=22 OOS) dan filter D1.
- Fee rata-rata 0,16R per trade; scalping M15 dengan stop sempit tidak layak di OKX.
- Skor makro sebagai prediktor arah harian tidak monoton → dipakai sebagai veto saja.

## Status 19 Okt 2026 14:15 WIB
Harga 4.193,5; H4 netral (EMA50 4.170,5 < EMA200 4.219,0); D1 turun (EMA20 4.186,2 < EMA50 4.243,9); makro +1 (DFII10 2,92→2,78; DXY −0,36%). Tidak ada setup v1. Range Asia 4.172,8–4.193,4, ATR H1 8,19.

## Next steps
- Paper trade 2–3 bulan di jurnal artifact sebelum uang sungguhan.
- Opsional: scheduled task per jam saat sesi London/NY + Telegram (bot sama dengan Radar BTC).
- Opsional: versi live GitHub Pages (fetch OKX langsung).
- Ide uji berikutnya (belum diuji): entry limit/maker, SL lebih lebar (3ATR/72j lebih baik di OOS: +0,076R net), filter ATR/harga.

Script: `claude/radar_xau_bt.py` (mesin backtest; data diambil ulang dari OKX/Yahoo/FRED).
