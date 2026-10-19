"""Evaluasi mingguan Radar XAU/USDT.

Menghitung ulang backtest aturan v1 di SELURUH data OKX XAU-USDT-SWAP (sejak listing Apr 2025 sampai minggu ini),
lalu membandingkannya dengan sinyal live di data/signals.json. Hasil -> data/backtest.json + ringkasan Telegram.

Tujuannya memantau, BUKAN mengganti parameter otomatis. Varian SL/TP di bawah sudah ditetapkan dari awal
(sama dengan backtest #1) supaya tidak terjadi curve fitting: perubahan aturan diputuskan manual setelah dievaluasi.
"""
import json, os, sys, time, urllib.parse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_xau as R

ROOT = R.ROOT
OUT = os.path.join(ROOT, "data", "backtest.json")
LIVE_START = pd.Timestamp("2026-10-19T09:00:00Z")      # radar Telegram mulai jalan
LISTING = pd.Timestamp("2025-04-09T00:00:00Z")
FEE_MK = 0.0004
GRID = [(sl, tp) for sl in (1.5, 2.0, 2.5, 3.0) for tp in (1.5, 2.0, 3.0)]

# ------------------------------------------------------------------ data
def fetch_15m_all():
    rows, after = {}, None
    while True:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={R.INST}&bar=15m&limit=100" + (f"&after={after}" if after else "")
        d = R.http_json(url)["data"]
        if not d: break
        for r in d:
            if r[8] == "1": rows[int(r[0])] = r
        after = d[-1][0]
        time.sleep(0.11)
    out = sorted(rows.values(), key=lambda r: int(r[0]))
    df = pd.DataFrame([[int(r[0]), *map(float, r[1:5])] for r in out], columns=["ts", "o", "h", "l", "c"])
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df.drop(columns="ts")

def yahoo_long(sym):
    q = urllib.parse.quote(sym, safe="")
    u = f"https://query1.finance.yahoo.com/v8/finance/chart/{q}?interval=1d&range=3y"
    err = None
    for url in (u, u.replace("query1", "query2"), R.YAHOO_PROXY + urllib.parse.quote(u, safe="")):
        try:
            r = R.http_json(url, tries=2)["chart"]["result"][0]
            s = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(r["timestamp"], unit="s", utc=True).normalize()).dropna()
            return s[~s.index.duplicated(keep="last")]
        except Exception as e:
            err = e
    raise RuntimeError(f"Yahoo {sym}: {err}")

def fred_long():
    import csv, io
    txt = R.http_text("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&cosd=2024-06-01")
    rows = list(csv.reader(io.StringIO(txt)))[1:]
    return pd.Series([pd.to_numeric(r[1], errors="coerce") for r in rows], index=pd.to_datetime([r[0] for r in rows], utc=True)).dropna()

# ------------------------------------------------------------------ mesin (vektor, identik dengan backtest/bt.py)
def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()

def build(m15, dxy, ry):
    h1, h4, d1 = resample(m15, "1h"), resample(m15, "4h"), resample(m15, "1D")
    h1, h4, d1 = R.prep(h1, h4, d1)
    h4s = h4[["trend"]].copy(); h4s.index = h4s.index + pd.Timedelta(hours=4)
    h1 = h1.join(h4s.reindex(h1.index, method="ffill"))
    d1s = d1[["dtrend"]].copy(); d1s.index = d1s.index + pd.Timedelta(days=1)
    h1 = h1.join(d1s.reindex(h1.index, method="ffill"))
    h1["hr"] = h1.index.hour; h1["dow"] = h1.index.dayofweek
    a = h1[h1.hr < 7]; asia = a.groupby(a.index.normalize()).agg(ah=("h", "max"), al=("l", "min"))
    h1 = h1.join(asia, on=h1.index.normalize())
    m = pd.DataFrame({"dxy": dxy, "ry": ry}).sort_index().ffill()
    m["dxy_ch5"] = m.dxy.pct_change(5) * 100; m["ry_ch5"] = m.ry.diff(5)
    m["macro"] = np.where(m.dxy_ch5 < -0.5, 1, np.where(m.dxy_ch5 > 0.5, -1, 0)) + np.where(m.ry_ch5 < -0.05, 1, np.where(m.ry_ch5 > 0.05, -1, 0))
    ms = m[["macro"]].copy(); ms.index = ms.index + pd.Timedelta(days=2); ms = ms[~ms.index.duplicated(keep="last")]
    h1 = h1.join(ms.reindex(h1.index.normalize(), method="ffill").set_index(h1.index))
    return h1

def breakout(h1):
    c = h1.c; t = h1.trend
    ok = (h1.dow < 5) & ~((h1.dow == 4) & (h1.hr >= 16)) & (h1.hr >= 7) & (h1.hr <= 15)
    lo = ok & (t == 1) & (c > h1.ah) & (c.shift() <= h1.ah)
    sh = ok & (t == -1) & (c < h1.al) & (c.shift() >= h1.al)
    return (lo.astype(int) - sh.astype(int)).fillna(0).astype(int)

def apply_filters(h1, sig, d1=True, macro=True):
    s = sig.copy()
    if d1: s = s.where(np.sign(h1.dtrend) == s, 0)
    if macro: s = s.where(~(np.sign(h1.macro) * s < 0), 0)
    return s.astype(int)

def simulate(m15, h1, sig, sl_atr=2.0, tp_r=2.0, max_bars=24):
    idx = m15.index; H, L, C, O = m15.h.values, m15.l.values, m15.c.values, m15.o.values
    pos = {t: i for i, t in enumerate(idx)}
    trades, busy = [], None
    for t in sig[sig != 0].index:
        et = t + pd.Timedelta(hours=1)
        if busy is not None and et < busy: continue
        i0 = pos.get(et)
        if i0 is None: continue
        d = int(sig.at[t]); a = h1.at[t, "atr"]
        if not np.isfinite(a) or a <= 0: continue
        e = O[i0]; risk = sl_atr * a; sl = e - d * risk; tp = e + d * tp_r * risk
        iend = i0 + max_bars * 4 - 1
        if iend >= len(C): break                       # trade belum selesai -> tidak dihitung
        res, ex = None, iend
        for i in range(i0, iend + 1):
            if (L[i] <= sl) if d == 1 else (H[i] >= sl): res, ex = -1.0, i; break
            if (H[i] >= tp) if d == 1 else (L[i] <= tp): res, ex = tp_r, i; break
        if res is None: res = d * (C[iend] - e) / risk
        trades.append(dict(t=t, dir=d, entry=e, risk=risk, r=res, r_net=res - R.FEE_RT * e / risk, r_mk=res - FEE_MK * e / risk,
                           hr=int(h1.at[t, "hr"]), d1=int(h1.at[t, "dtrend"]), macro=int(h1.at[t, "macro"]) if np.isfinite(h1.at[t, "macro"]) else 0))
        busy = idx[ex] + pd.Timedelta(minutes=15)
    return pd.DataFrame(trades)

def stats(tr, col="r_net"):
    if tr is None or len(tr) == 0: return dict(n=0)
    r = tr[col]; eq = r.cumsum(); dd = float((eq - eq.cummax()).min())
    neg = -r[r < 0].sum()
    return dict(n=int(len(tr)), win=round(float((tr.r > 0).mean() * 100), 1), avg=round(float(r.mean()), 3),
                tot=round(float(r.sum()), 2), pf=(round(float(r[r > 0].sum() / neg), 2) if neg > 0 else None), dd=round(dd, 2),
                cur_dd=round(float(eq.iloc[-1] - eq.max()), 2))

# ------------------------------------------------------------------ evaluasi
def evaluate_all(m15, h1, live):
    now = m15.index[-1]
    base = breakout(h1); v1 = apply_filters(h1, base)
    tr = simulate(m15, h1, v1)
    per = {"semua": tr, "6 bulan": tr[tr.t >= now - pd.Timedelta(days=182)], "3 bulan": tr[tr.t >= now - pd.Timedelta(days=91)],
           "sejak live": tr[tr.t >= LIVE_START - pd.Timedelta(hours=1)]}
    out = {"periods": {k: stats(v) for k, v in per.items()}}
    out["maker_avg"] = round(float(tr.r_mk.mean()), 3) if len(tr) else None
    q = tr.groupby(tr.t.dt.tz_localize(None).dt.to_period("Q")).r_net.agg(["size", "sum"])
    out["quarters"] = [dict(q=str(p), n=int(r["size"]), tot=round(float(r["sum"]), 2)) for p, r in q.iterrows()]
    out["side"] = {"long": stats(tr[tr.dir == 1]), "short": stats(tr[tr.dir == -1])}
    out["session"] = {"London 14–18 WIB": stats(tr[tr.hr <= 11]), "NY 19–22 WIB": stats(tr[tr.hr >= 12])}

    # atribusi filter: semua breakout searah H4, dikelompokkan menurut apakah filter meloloskannya
    raw = simulate(m15, h1, base)
    six = now - pd.Timedelta(days=182)
    def grp(df):
        d1_ok = df.d1 == df.dir; mac_ok = ~(np.sign(df.macro) * df.dir < 0)
        return {"lolos semua filter": stats(df[d1_ok & mac_ok]), "diblok D1": stats(df[~d1_ok]), "diblok makro": stats(df[d1_ok & ~mac_ok])}
    out["filters"] = {"semua": grp(raw), "6 bulan": grp(raw[raw.t >= six])}

    # sensitivitas (varian tetap, hanya untuk dipantau)
    sens = []
    for sl, tp in GRID:
        t2 = simulate(m15, h1, v1, sl, tp, 24)
        sens.append(dict(sl=sl, tp=tp, semua=stats(t2).get("avg"), n=len(t2), enam=stats(t2[t2.t >= six]).get("avg")))
    out["sensitivity"] = sens

    # kecocokan live vs backtest
    bt_live = tr[tr.t >= LIVE_START - pd.Timedelta(hours=1)]
    lv = [s for s in live if s.get("status") in ("TP", "SL", "WAKTU")]
    bset = {(pd.Timestamp(t).isoformat(), int(d)): r for t, d, r in zip(bt_live.t, bt_live.dir, bt_live.r)}
    lset = {(pd.Timestamp(s["t"]).isoformat(), int(s["dir"])): s["r"] for s in lv}
    last_closed = max([pd.Timestamp(s["exit_t"]) for s in lv], default=None)
    only_bt = [k for k in bset if k not in lset and (last_closed is None or pd.Timestamp(k[0]) <= last_closed)]
    out["live"] = dict(selesai=len(lv), cocok=sum(1 for k in lset if k in bset), hanya_live=[k[0] for k in lset if k not in bset],
                       hanya_backtest=[k[0] for k in only_bt], beda_hasil=[k[0] for k in lset if k in bset and abs(lset[k] - bset[k]) > 0.05],
                       stats=stats(pd.DataFrame([dict(r=s["r"], r_net=s["r_net"]) for s in lv])) if lv else dict(n=0))

    # tanda peringatan
    flags = []
    p3, p6, pa = out["periods"]["3 bulan"], out["periods"]["6 bulan"], out["periods"]["semua"]
    if p6.get("n", 0) >= 10 and p6["avg"] < 0: flags.append(f"Rata-rata 6 bulan terakhir negatif ({p6['avg']:+.2f}R, n={p6['n']}). Edge mungkin hilang.")
    elif p3.get("n", 0) >= 6 and p3["avg"] < 0: flags.append(f"Rata-rata 3 bulan terakhir negatif ({p3['avg']:+.2f}R, n={p3['n']}). Pantau, belum tentu permanen.")
    if pa.get("n") and pa["cur_dd"] <= pa["dd"] + 1e-9 and pa["cur_dd"] < -3: flags.append(f"Sedang di drawdown terdalam sepanjang data ({pa['cur_dd']:.1f}R).")
    fm = out["filters"]["6 bulan"]["diblok makro"]; fp = out["filters"]["6 bulan"]["lolos semua filter"]
    if fm.get("n", 0) >= 8 and fp.get("n", 0) >= 8 and fm["avg"] > fp["avg"]: flags.append(f"Veto makro 6 bulan terakhir tidak membantu: yang diblok {fm['avg']:+.2f}R vs yang lolos {fp['avg']:+.2f}R.")
    fd = out["filters"]["6 bulan"]["diblok D1"]
    if fd.get("n", 0) >= 8 and fp.get("n", 0) >= 8 and fd["avg"] > fp["avg"]: flags.append(f"Filter D1 6 bulan terakhir tidak membantu: yang diblok {fd['avg']:+.2f}R vs yang lolos {fp['avg']:+.2f}R.")
    L = out["live"]
    if L["hanya_live"] or L["hanya_backtest"] or L["beda_hasil"]:
        flags.append(f"Sinyal live tidak sama dengan backtest (hanya live {len(L['hanya_live'])}, hanya backtest {len(L['hanya_backtest'])}, hasil beda {len(L['beda_hasil'])}). Perlu dicek.")
    out["flags"] = flags
    return out, tr

def msg(rep):
    p = rep["periods"]; L = rep["live"]
    def line(k):
        s = p[k]
        return f"{k}: {s['n']} sinyal · win {s['win']:.0f}% · {s['avg']:+.2f}R/sinyal · total {s['tot']:+.1f}R" if s.get("n") else f"{k}: belum ada sinyal"
    f = rep["filters"]["6 bulan"]
    def fl(k): s = f[k]; return f"{s['avg']:+.2f}R (n={s['n']})" if s.get("n") else "–"
    g = pd.Timestamp(rep["generated"]).tz_convert(R.WIB)
    txt = (f"<b>[XAU] Evaluasi mingguan — {g.day} {R.BULAN[g.month-1]} {g.year}</b>\n"
           f"Backtest ulang aturan v1, data OKX s/d {R.wib(rep['data_to'])} (setelah fee)\n\n"
           + "\n".join(line(k) for k in ("semua", "6 bulan", "3 bulan", "sejak live")) +
           f"\n\nFilter 6 bulan: lolos {fl('lolos semua filter')} · diblok D1 {fl('diblok D1')} · diblok makro {fl('diblok makro')}\n"
           f"Live: {L['selesai']} sinyal selesai, {L['cocok']} cocok dengan backtest\n\n")
    txt += ("<b>Perlu perhatian:</b>\n" + "\n".join("• " + x for x in rep["flags"])) if rep["flags"] else "Tidak ada tanda peringatan."
    return txt

def main():
    gen = pd.Timestamp.now(tz="UTC")
    m15 = fetch_15m_all()
    dxy = yahoo_long("DX-Y.NYB"); ry = fred_long()
    h1 = build(m15, dxy, ry)
    try: live = json.load(open(R.STATE))["signals"]
    except Exception: live = []
    rep, tr = evaluate_all(m15, h1, live)
    rep.update(generated=gen.isoformat(), data_from=m15.index[0].isoformat(), data_to=(m15.index[-1] + pd.Timedelta(minutes=15)).isoformat(),
               candles=int(len(m15)), rules="v1: breakout range Asia + tren H4 + D1 + veto makro, SL 2xATR14 H1, TP 2R, 24 jam, fee taker 0,05%/sisi",
               equity=[dict(t=t.isoformat()[:16], r=round(float(r), 3)) for t, r in zip(tr.t, tr.r_net)])
    R.save(OUT, rep)
    print(msg(rep))
    if os.environ.get("SEND_TELEGRAM", "1") == "1": R.send(msg(rep))

if __name__ == "__main__":
    main()
