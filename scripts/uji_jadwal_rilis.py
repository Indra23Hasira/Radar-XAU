"""Uji historis: apakah sinyal v1 di dekat jam rilis data AS berdampak tinggi (3 bintang) hasilnya berbeda?

Jadwal rilis diambil dari FRED API (butuh secret FRED_API_KEY): CPI, PPI, NFP, GDP, PCE, Retail Sales (08:30 ET)
ditambah keputusan FOMC (14:00 ET, tanggal rapat ditulis manual). Data harga: OKX XAU-USDT-SWAP M15 sejak listing.
Proyeksi konsensus TIDAK tersedia historis, jadi uji ini hanya soal WAKTU rilis, bukan aktual vs proyeksi.
Aturan yang diuji sudah ditetapkan sebelum melihat hasil (R1, R2) supaya tidak mencari-cari.
Output: data/uji-rilis.json
"""
import json, os, sys
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radar_xau as R
import evaluasi_mingguan as E

OUT = os.path.join(R.ROOT, "data", "uji-rilis.json")
ET = ZoneInfo("America/New_York")
RELEASES = {10: "CPI", 46: "PPI", 50: "NFP", 53: "GDP", 54: "PCE", 9: "Retail Sales"}
FOMC = ["2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28"]

def fred_api(path, **kw):
    key = os.environ["FRED_API_KEY"]
    q = "&".join(f"{k}={v}" for k, v in kw.items())
    return R.http_json(f"https://api.stlouisfed.org/fred/{path}?api_key={key}&file_type=json&{q}")

def events(start, end):
    rows, names = [], {}
    for rid, lab in RELEASES.items():
        names[lab] = fred_api("release", release_id=rid)["releases"][0]["name"]
        d = fred_api("release/dates", release_id=rid, realtime_start=start, realtime_end=end,
                     include_release_dates_with_no_data="false", sort_order="asc", limit=1000)
        for x in d["release_dates"]:
            t = pd.Timestamp(x["date"] + " 08:30", tz=ET).tz_convert("UTC")
            rows.append(dict(t=t, name=lab))
    for d in FOMC:
        rows.append(dict(t=pd.Timestamp(d + " 14:00", tz=ET).tz_convert("UTC"), name="FOMC"))
    ev = pd.DataFrame(rows).sort_values("t")
    ev = ev[(ev.t >= pd.Timestamp(start, tz="UTC")) & (ev.t <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
    # beberapa rilis bisa di jam yang sama (mis. GDP + PCE) -> gabungkan
    ev = ev.groupby("t").name.apply(lambda s: " + ".join(sorted(set(s)))).reset_index()
    return ev, names

def simulate_exit(m15, h1, sig, sl_atr=2.0, tp_r=2.0, max_bars=24):
    idx = m15.index; H, L, C, O = m15.h.values, m15.l.values, m15.c.values, m15.o.values
    pos = {t: i for i, t in enumerate(idx)}
    out, busy = [], None
    for t in sig[sig != 0].index:
        et = t + pd.Timedelta(hours=1)
        if busy is not None and et < busy: continue
        i0 = pos.get(et)
        if i0 is None: continue
        d = int(sig.at[t]); a = h1.at[t, "atr"]
        if not np.isfinite(a) or a <= 0: continue
        e = O[i0]; risk = sl_atr * a; sl = e - d * risk; tp = e + d * tp_r * risk
        iend = i0 + max_bars * 4 - 1
        if iend >= len(C): break
        res, ex = None, iend
        for i in range(i0, iend + 1):
            if (L[i] <= sl) if d == 1 else (H[i] >= sl): res, ex = -1.0, i; break
            if (H[i] >= tp) if d == 1 else (L[i] <= tp): res, ex = tp_r, i; break
        if res is None: res = d * (C[iend] - e) / risk
        out.append(dict(t=t, entry_t=et, exit_t=idx[ex] + pd.Timedelta(minutes=15), dir=d, r=res, r_net=res - R.FEE_RT * e / risk))
        busy = idx[ex] + pd.Timedelta(minutes=15)
    return pd.DataFrame(out)

def tag(tr, ev):
    T = ev.t.values
    def any_between(a, b): return bool(((T > np.datetime64(a.tz_convert(None))) & (T <= np.datetime64(b.tz_convert(None)))).any())
    tr = tr.copy()
    # A: rilis terjadi dalam 1 jam sebelum candle pemicu dibuka s/d saat candle pemicu tutup (breakout kemungkinan dipicu rilis)
    tr["A"] = [any_between(t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1)) for t in tr.t]
    # B: rilis terjadi saat posisi berjalan
    tr["B"] = [any_between(a, b) for a, b in zip(tr.entry_t, tr.exit_t)]
    # B4: rilis dalam 4 jam pertama setelah entry (sudah diketahui dari kalender saat sinyal muncul)
    tr["B4"] = [any_between(a, a + pd.Timedelta(hours=4)) for a in tr.entry_t]
    return tr

def reaction(m15, ev):
    """Gerak absolut emas 15 menit & 1 jam setelah rilis vs jam yang sama di hari kerja tanpa rilis."""
    c = m15.c; o = m15.o
    def move(t, mins):
        if t not in o.index: return np.nan
        end = t + pd.Timedelta(minutes=mins - 15)
        if end not in c.index: return np.nan
        return abs(c.at[end] / o.at[t] - 1) * 100
    ev_days = set(ev.t.dt.normalize())
    rows = []
    for t, name in zip(ev.t, ev.name):
        base = [move(t.normalize() - pd.Timedelta(days=k) + (t - t.normalize()), 60) for k in range(1, 30)
                if (t - pd.Timedelta(days=k)).dayofweek < 5 and (t.normalize() - pd.Timedelta(days=k)) not in ev_days]
        base = [b for b in base if np.isfinite(b)][:10]
        rows.append(dict(t=t, name=name, m15=move(t, 15), m60=move(t, 60), base60=np.median(base) if base else np.nan))
    df = pd.DataFrame(rows).dropna()
    df["ratio"] = df.m60 / df.base60
    out = {}
    for key, g in [("semua", df)] + [(n, df[df.name.str.contains(n, regex=False)]) for n in ["CPI", "NFP", "PCE", "GDP", "PPI", "Retail Sales", "FOMC"]]:
        if len(g): out[key] = dict(n=int(len(g)), gerak_15m=round(float(g.m15.median()), 3), gerak_1j=round(float(g.m60.median()), 3),
                                    normal_1j=round(float(g.base60.median()), 3), kali_normal=round(float(g.ratio.median()), 2))
    return out

def grp(tr):
    return {"pemicu dekat rilis (A)": E.stats(tr[tr.A]), "bukan A": E.stats(tr[~tr.A]),
            "rilis saat posisi jalan (B)": E.stats(tr[tr.B]), "tanpa rilis selama posisi": E.stats(tr[~tr.B]),
            "rilis dalam 4 jam setelah entry (B4)": E.stats(tr[tr.B4]), "bukan B4": E.stats(tr[~tr.B4]),
            "bebas rilis (bukan A & bukan B)": E.stats(tr[~tr.A & ~tr.B])}

def main():
    m15 = E.fetch_15m_all()
    start, end = m15.index[0].date().isoformat(), m15.index[-1].date().isoformat()
    ev, names = events(start, end)
    dxy = E.yahoo_long("DX-Y.NYB"); ry = E.fred_long()
    h1 = E.build(m15, dxy, ry)
    base = E.breakout(h1); v1 = E.apply_filters(h1, base)
    tr = tag(simulate_exit(m15, h1, v1), ev)
    raw = tag(simulate_exit(m15, h1, base), ev)
    now = m15.index[-1]; six = now - pd.Timedelta(days=182)
    rules = {"v1 (sekarang)": E.stats(tr),
             "R1: lewati sinyal yang pemicunya dekat rilis": E.stats(tr[~tr.A]),
             "R2: lewati sinyal kalau ada rilis dalam 4 jam setelah entry": E.stats(tr[~tr.B4]),
             "R1 + R2": E.stats(tr[~tr.A & ~tr.B4])}
    split = {"2025": E.stats(tr[tr.t.dt.year == 2025]), "2025 tanpa A": E.stats(tr[(tr.t.dt.year == 2025) & ~tr.A]),
             "2026": E.stats(tr[tr.t.dt.year == 2026]), "2026 tanpa A": E.stats(tr[(tr.t.dt.year == 2026) & ~tr.A])}
    rep = dict(generated=pd.Timestamp.now(tz="UTC").isoformat(), data_from=start, data_to=end, release_names=names,
               n_events=int(len(ev)), events_per_month=round(len(ev) / ((m15.index[-1] - m15.index[0]).days / 30.4), 1),
               reaction=reaction(m15, ev), v1_groups=grp(tr), breakout_groups=grp(raw), rules=rules, split=split,
               v1_trades=[dict(t=str(r.t), dir=int(r.dir), r_net=round(float(r.r_net), 3), A=bool(r.A), B=bool(r.B), B4=bool(r.B4)) for r in tr.itertuples()],
               events=[dict(t=str(t), name=n) for t, n in zip(ev.t, ev.name)])
    R.save(OUT, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("v1_trades", "events")}, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
