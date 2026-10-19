"""Backtest setup intraday XAU/USDT (OKX XAU-USDT-SWAP).
Sinyal dari candle H1 yang sudah tutup, entry di open H1 berikutnya,
SL/TP dicek di candle M15 (kalau SL & TP kena di candle M15 yang sama -> dianggap SL).
Satu posisi per setup pada satu waktu (tidak ada sampel tumpang tindih)."""
import json, numpy as np, pandas as pd

FEE_RT = 0.0010   # taker 0.05% x2 (konservatif)
FEE_MK = 0.0004   # maker 0.02% x2

def load_15m():
    r = json.load(open("data/okx_xau_15m.json"))
    df = pd.DataFrame([[int(x[0]), *map(float, x[1:6])] for x in r], columns="ts o h l c v".split())
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df.drop(columns="ts")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+up/dn)
def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h-df.l, (df.h-pc).abs(), (df.l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()

def build():
    m15 = load_15m()
    h1 = resample(m15, "1h")
    h4 = resample(m15, "4h")
    d1 = resample(m15, "1D")
    # --- H4 trend (diketahui setelah candle H4 tutup) ---
    h4["ema50"] = ema(h4.c, 50); h4["ema200"] = ema(h4.c, 200)
    h4["ema50_slope"] = h4.ema50 - h4.ema50.shift(3)
    h4["trend"] = np.where((h4.c > h4.ema50) & (h4.ema50 > h4.ema200), 1,
                  np.where((h4.c < h4.ema50) & (h4.ema50 < h4.ema200), -1, 0))
    # geser: nilai H4 berlaku mulai jam setelah candle H4 itu tutup
    h4s = h4[["trend"]].copy(); h4s.index = h4s.index + pd.Timedelta(hours=4)
    h1 = h1.join(h4s.reindex(h1.index, method="ffill"))
    # --- D1 trend ---
    d1["ema20"] = ema(d1.c, 20); d1["ema50"] = ema(d1.c, 50)
    d1["dtrend"] = np.where(d1.ema20 > d1.ema50, 1, -1)
    d1s = d1[["dtrend"]].copy(); d1s.index = d1s.index + pd.Timedelta(days=1)
    h1 = h1.join(d1s.reindex(h1.index, method="ffill"))
    # --- H1 indikator ---
    h1["ema20"] = ema(h1.c, 20); h1["ema50"] = ema(h1.c, 50)
    h1["rsi"] = rsi(h1.c); h1["atr"] = atr(h1)
    h1["hh20"] = h1.h.rolling(20).max(); h1["ll20"] = h1.l.rolling(20).min()
    h1["hr"] = h1.index.hour; h1["dow"] = h1.index.dayofweek
    # Asia range (00-07 UTC) per hari
    day = h1.index.normalize()
    asia = h1[(h1.hr < 7)].groupby(h1[(h1.hr < 7)].index.normalize()).agg(ah=("h","max"), al=("l","min"))
    h1 = h1.join(asia, on=day)
    # --- makro harian (nilai hari sebelumnya, tanpa look-ahead) ---
    mac = json.load(open("data/macro_daily.json"))
    def ser(k):
        s = pd.Series(mac[k]["c"], index=pd.to_datetime(mac[k]["t"], unit="s", utc=True).normalize()).dropna()
        return s[~s.index.duplicated(keep="last")]
    dxy = ser("DX-Y.NYB"); vix = ser("^VIX")
    ry = pd.read_csv("data/dfii10.csv"); ry = pd.Series(pd.to_numeric(ry.DFII10, errors="coerce").values, index=pd.to_datetime(ry.observation_date, utc=True)).dropna()
    m = pd.DataFrame({"dxy": dxy, "ry": ry, "vix": vix}).sort_index().ffill()
    m["dxy_ch5"] = m.dxy.pct_change(5) * 100
    m["ry_ch5"] = m.ry.diff(5)
    m["s_dxy"] = np.where(m.dxy_ch5 < -0.5, 1, np.where(m.dxy_ch5 > 0.5, -1, 0))
    m["s_ry"] = np.where(m.ry_ch5 < -0.05, 1, np.where(m.ry_ch5 > 0.05, -1, 0))
    m["macro"] = m.s_dxy + m.s_ry
    # data tanggal D tersedia setelah tutup D (FRED: rilis D+1) -> berlaku mulai D+2 supaya aman
    ms = m[["macro", "s_dxy", "s_ry", "dxy_ch5", "ry_ch5"]].copy(); ms.index = ms.index + pd.Timedelta(days=2)
    ms = ms[~ms.index.duplicated(keep="last")]
    h1 = h1.join(ms.reindex(h1.index.normalize(), method="ffill").set_index(h1.index))
    return m15, h1

# ---------------- setup ----------------
def signals(h1, name):
    """return Series arah (+1 long / -1 short / 0) per bar H1 (sinyal di close bar itu)"""
    c, o, h, l = h1.c, h1.o, h1.h, h1.l
    weekday = h1.dow < 5
    fri_late = (h1.dow == 4) & (h1.hr >= 16)
    ok = weekday & ~fri_late
    t = h1.trend
    if name == "baseline_trend":           # masuk tiap bar searah tren H4 (pembanding)
        sig = t
    elif name == "pullback_ema":           # tren H4, low sentuh zona EMA50 H1 (±0.3 ATR), close kembali di atas EMA20
        lo = (t == 1) & (l <= h1.ema50 + 0.3*h1.atr) & (c > h1.ema20) & (c > o)
        sh = (t == -1) & (h >= h1.ema50 - 0.3*h1.atr) & (c < h1.ema20) & (c < o)
        sig = lo.astype(int) - sh.astype(int)
    elif name == "fib_pullback":           # tren H4, harga retrace 50-78.6% dari swing 20 bar, candle rejection
        rng = h1.hh20 - h1.ll20
        f50u, f786u = h1.hh20 - 0.5*rng, h1.hh20 - 0.786*rng
        f50d, f786d = h1.ll20 + 0.5*rng, h1.ll20 + 0.786*rng
        lo = (t == 1) & (l <= f50u) & (l >= f786u - 0.2*h1.atr) & (c > o) & (rng > 2*h1.atr)
        sh = (t == -1) & (h >= f50d) & (h <= f786d + 0.2*h1.atr) & (c < o) & (rng > 2*h1.atr)
        sig = lo.astype(int) - sh.astype(int)
    elif name == "asia_breakout":          # close H1 tembus range Asia saat London/NY (07-16 UTC), searah tren H4
        win = (h1.hr >= 7) & (h1.hr <= 15)
        lo = win & (t == 1) & (c > h1.ah) & (c.shift() <= h1.ah)
        sh = win & (t == -1) & (c < h1.al) & (c.shift() >= h1.al)
        sig = lo.astype(int) - sh.astype(int)
    elif name == "asia_breakout_any":      # sama tapi tanpa filter tren
        win = (h1.hr >= 7) & (h1.hr <= 15)
        lo = win & (c > h1.ah) & (c.shift() <= h1.ah)
        sh = win & (c < h1.al) & (c.shift() >= h1.al)
        sig = lo.astype(int) - sh.astype(int)
    elif name == "rsi_reset":              # tren H4, RSI H1 turun ke 35-45 lalu naik lagi (long) / mirror
        lo = (t == 1) & (h1.rsi.shift() < 42) & (h1.rsi > h1.rsi.shift()) & (h1.rsi.shift() > 30)
        sh = (t == -1) & (h1.rsi.shift() > 58) & (h1.rsi < h1.rsi.shift()) & (h1.rsi.shift() < 70)
        sig = lo.astype(int) - sh.astype(int)
    elif name == "mean_rev":               # kontra-tren: RSI H1 ekstrem <25 long / >75 short
        sig = (h1.rsi < 25).astype(int) - (h1.rsi > 75).astype(int)
    else:
        raise ValueError(name)
    return (sig.fillna(0) * ok).astype(int)

def simulate(m15, h1, sig, sl_atr=1.5, tp_r=2.0, max_bars=24, sessions=None):
    """SL = sl_atr x ATR H1, TP = tp_r x jarak SL, keluar paksa setelah max_bars jam."""
    idx15 = m15.index; H = m15.h.values; L = m15.l.values; C = m15.c.values; O = m15.o.values
    pos15 = {t: i for i, t in enumerate(idx15)}
    trades = []; busy_until = None
    sig_idx = sig[sig != 0].index
    for t in sig_idx:
        if sessions is not None and h1.at[t, "hr"] not in sessions: continue
        entry_t = t + pd.Timedelta(hours=1)
        if busy_until is not None and entry_t < busy_until: continue
        i0 = pos15.get(entry_t)
        if i0 is None: continue
        d = int(sig.at[t]); a = h1.at[t, "atr"]
        if not np.isfinite(a) or a <= 0: continue
        e = O[i0]; risk = sl_atr * a
        sl = e - d*risk; tp = e + d*tp_r*risk
        iend = min(i0 + max_bars*4, len(C)) - 1
        res = None; exit_i = iend; mae = 0.0
        for i in range(i0, iend+1):
            adv = (e - L[i]) if d == 1 else (H[i] - e)
            mae = max(mae, adv)
            hit_sl = (L[i] <= sl) if d == 1 else (H[i] >= sl)
            hit_tp = (H[i] >= tp) if d == 1 else (L[i] <= tp)
            if hit_sl: res = -1.0; exit_i = i; break
            if hit_tp: res = tp_r; exit_i = i; break
        if res is None: res = d*(C[iend]-e)/risk
        fee_r = FEE_RT*e/risk; fee_mk = FEE_MK*e/risk
        trades.append(dict(t=t, dir=d, entry=e, risk=risk, r_gross=res, r_net=res-fee_r, r_maker=res-fee_mk,
                           mae_r=mae/risk, macro=h1.at[t, "macro"], dtrend=h1.at[t, "dtrend"], hr=h1.at[t, "hr"],
                           bars=(exit_i-i0+1)/4))
        busy_until = idx15[exit_i] + pd.Timedelta(minutes=15)
    return pd.DataFrame(trades)

def stats(tr, col="r_net"):
    if len(tr) == 0: return dict(n=0)
    r = tr[col]; eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    pf = r[r > 0].sum() / max(1e-9, -r[r < 0].sum())
    return dict(n=len(tr), win=round((tr.r_gross > 0).mean()*100, 1), avgR=round(r.mean(), 3),
                totR=round(r.sum(), 1), pf=round(pf, 2), maxDD_R=round(dd, 1))
