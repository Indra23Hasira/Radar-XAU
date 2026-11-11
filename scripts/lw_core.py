"""Strategi kedua [XAU-LW] — volatility breakout Larry Williams + filter radar (forward test sejak 11 Nov 2026).

Aturan (hari = hari UTC, 07:00–06:59 WIB):
  garis BUY  = open hari ini + 0,5 x range (high-low) kemarin
  garis SELL = open hari ini - 0,5 x range kemarin
  sentuhan PERTAMA menentukan arah. Arah hanya diambil kalau searah tren D1 (EMA20 vs EMA50 harian kemarin)
  DAN skor makro tidak berlawanan. Kalau sentuhan pertama arahnya diblok -> tidak ada transaksi hari itu.
  SL di harga open, keluar di close hari (07:00 WIB besok). Kalau BUY & SELL kena di candle M15 yang sama -> dilewati.
  Senin dicatat sebagai EKSPERIMEN (range Minggu dari perpetual akhir pekan terlalu kecil; hasil backtest Senin tidak dipercaya).
Backtest (19 bln, Sel–Jum, dengan filter): 84 transaksi, +0,19R/transaksi setelah fee, PF 1,45.
Satu fungsi `simulate_day` dipakai backtest DAN live supaya hasilnya identik.
"""
import json, os
import numpy as np, pandas as pd
import radar_xau as R

K = 0.5
STATE = os.path.join(R.ROOT, "data", "lw.json")
LIVE_START = pd.Timestamp("2026-11-12T00:00:00Z")
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

def levels(o, rng):
    return o + K * rng, o - K * rng

def allowed_dirs(dtrend, macro):
    out = set()
    for d in (1, -1):
        if dtrend == d and not (np.sign(macro) * d < 0): out.add(d)
    return out

def simulate_day(bars, o, rng, allowed, complete):
    """bars: candle M15 hari itu yang SUDAH tutup (urut). Return dict status."""
    up, dn = levels(o, rng)
    d, entry, entry_t = 0, None, None
    for t, x in bars.iterrows():
        if d == 0:
            hu, hd = x.h >= up, x.l <= dn
            if hu and hd: return dict(status="AMBIGU", up=up, dn=dn)
            if not (hu or hd): continue
            d = 1 if hu else -1
            entry = max(up, x.o) if d == 1 else min(dn, x.o)
            entry_t = t
            if d not in allowed: return dict(status="DIBLOK", dir=d, up=up, dn=dn, entry_t=str(t))
            if (d == 1 and x.l <= o and x.c < o) or (d == -1 and x.h >= o and x.c > o):
                return dict(status="SL", dir=d, entry=entry, entry_t=str(t), exit=o, exit_t=str(t + pd.Timedelta(minutes=15)), r=-1.0, up=up, dn=dn)
            continue
        if (d == 1 and x.l <= o) or (d == -1 and x.h >= o):
            return dict(status="SL", dir=d, entry=entry, entry_t=str(entry_t), exit=o, exit_t=str(t + pd.Timedelta(minutes=15)), r=-1.0, up=up, dn=dn)
    if d == 0: return dict(status="TUTUP_TANPA_SENTUHAN" if complete else "MENUNGGU", up=up, dn=dn)
    if complete:
        c = float(bars.c.iloc[-1])
        return dict(status="CLOSE", dir=d, entry=entry, entry_t=str(entry_t), exit=c, exit_t=str(bars.index[-1] + pd.Timedelta(minutes=15)),
                    r=d * (c - entry) / abs(entry - o), up=up, dn=dn)
    return dict(status="AKTIF", dir=d, entry=entry, entry_t=str(entry_t), up=up, dn=dn)

def r_net(res, o):
    return res["r"] - R.FEE_RT * res["entry"] / abs(res["entry"] - o)

def day_filter(d1, mrows, day):
    """Tren D1 & makro yang berlaku untuk hari UTC `day` (sama dengan radar v1)."""
    prev = d1[d1.index + pd.Timedelta(days=1) <= day]
    mr = R.macro_for(mrows, day)
    return int(prev.dtrend.iloc[-1]), (mr["macro"] if mr else 0), mr

def backtest(m15, d1, mrows):
    """m15: candle M15 tutup (o,h,l,c), d1: candle harian UTC yang sudah di-prep (dtrend)."""
    days = m15.groupby(m15.index.floor("D"))
    dd = days.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), n=("o", "size"))
    out = []
    keys = list(dd.index)
    for i in range(1, len(keys)):
        day, prev = keys[i], keys[i - 1]
        if day.dayofweek >= 5 or dd.at[day, "n"] < 96 or (day - prev).days != 1: continue
        rng = dd.at[prev, "h"] - dd.at[prev, "l"]; o = dd.at[day, "o"]
        if rng <= 0: continue
        try: dtr, mac, _ = day_filter(d1, mrows, day)
        except Exception: continue
        res = simulate_day(days.get_group(day), o, rng, allowed_dirs(dtr, mac), True)
        if res["status"] in ("SL", "CLOSE"):
            out.append(dict(day=day, dir=res["dir"], r=res["r"], r_net=r_net(res, o), senin=day.dayofweek == 0,
                            risk_pct=abs(res["entry"] - o) / res["entry"] * 100))
    return pd.DataFrame(out)

# ------------------------------------------------------------------ live
def fmt(x): return f"{x:,.1f}"

def live(d1_all, mrows, now):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"days": []}
    days = {x["day"]: x for x in st["days"]}
    today = now.floor("D")
    changed = False
    for day in (today - pd.Timedelta(days=1), today):
        if day.dayofweek >= 5 or day < LIVE_START: continue
        key = day.date().isoformat()
        rec = days.get(key)
        if rec and rec.get("final"): continue
        d1 = d1_all[d1_all.index < day]
        if not len(d1) or d1.index[-1] != day - pd.Timedelta(days=1): continue
        rng = float(d1.h.iloc[-1] - d1.l.iloc[-1])
        m15 = R.okx_15m_since(day)
        bars = m15[(m15.confirm == "1") & (m15.index < day + pd.Timedelta(days=1))]
        if not len(bars) or bars.index[0] != day: continue
        o = float(bars.o.iloc[0])
        dtr, mac, mr = day_filter(d1_all, mrows, day)
        allowed = allowed_dirs(dtr, mac)
        complete = len(bars) >= 96
        res = simulate_day(bars, o, rng, allowed, complete)
        if rec is None:
            rec = dict(day=key, open=o, range=rng, up=res["up"], dn=res["dn"], dtrend=dtr, macro=mac, allowed=sorted(allowed),
                       senin=day.dayofweek == 0, sent=[])
            days[key] = rec; changed = True
        prev_status = rec.get("status")
        rec.update({k: v for k, v in res.items() if k not in ("up", "dn")})
        if res["status"] in ("SL", "CLOSE"): rec["r_net"] = round(r_net(res, o), 3); rec["r"] = round(res["r"], 3)
        if res["status"] in ("SL", "CLOSE", "DIBLOK", "AMBIGU", "TUTUP_TANPA_SENTUHAN"): rec["final"] = True
        if rec.get("status") != prev_status: changed = True
        tag = "[XAU-LW]" + (" EKSPERIMEN SENIN" if rec["senin"] else "")
        # pesan 1: rencana pagi
        if "plan" not in rec["sent"] and day == today:
            if allowed:
                d = next(iter(allowed)); lvl = rec["up"] if d == 1 else rec["dn"]; other = rec["dn"] if d == 1 else rec["up"]
                risk = abs(lvl - o); ct = int(R.RISK_USDT / risk / R.CT_VAL)
                txt = (f"<b>{tag} Rencana {HARI[day.dayofweek]} {day.day} {R.BULAN[day.month-1]}</b>\n"
                       f"Arah diizinkan: <b>{'BUY' if d == 1 else 'SELL'}</b> (D1 {'naik' if dtr == 1 else 'turun'}, makro {R.sgn(mac)})\n"
                       f"{'BUY STOP' if d == 1 else 'SELL STOP'}: <b>{fmt(lvl)}</b> · SL {fmt(o)} (jarak {risk:.1f})\n"
                       f"Batal kalau harga menyentuh {fmt(other)} lebih dulu. Tutup posisi besok 07:00 WIB.\n"
                       f"Ukuran per {R.RISK_USDT:.0f} USDT risiko: {ct*R.CT_VAL:.3f} XAU = {ct} kontrak\n"
                       + ("<i>Senin = eksperimen, tidak dihitung di forward test utama.</i>" if rec["senin"] else "<i>Forward test — belum untuk uang sungguhan.</i>"))
            else:
                why = "D1 & makro tidak searah" if dtr * mac < 0 else "arah dibatasi filter"
                txt = f"<b>{tag} {HARI[day.dayofweek]} {day.day} {R.BULAN[day.month-1]}: skip</b>\nTidak ada arah yang diizinkan ({why}: D1 {'naik' if dtr == 1 else 'turun'}, makro {R.sgn(mac)})."
                rec["final"] = True; rec["status"] = "SKIP"
            if R.send(txt): rec["sent"].append("plan"); changed = True
        # pesan 2: entry
        if rec.get("entry") and rec.get("dir") in allowed and "entry" not in rec["sent"] and rec.get("status") in ("AKTIF", "SL", "CLOSE"):
            txt = (f"<b>{tag} ENTRY {'BUY' if rec['dir'] == 1 else 'SELL'}</b> di {fmt(rec['entry'])} ({R.wib(rec['entry_t'], False)})\n"
                   f"SL {fmt(o)} · tutup besok 07:00 WIB")
            if R.send(txt): rec["sent"].append("entry"); changed = True
        # pesan batal: arah yang diblok tersentuh lebih dulu
        if rec.get("status") == "DIBLOK" and "plan" in rec["sent"] and rec.get("allowed") and "batal" not in rec["sent"]:
            lvl = rec["up"] if rec["dir"] == 1 else rec["dn"]
            txt = f"<b>{tag} BATAL hari ini</b>\nGaris {'BUY' if rec['dir'] == 1 else 'SELL'} {fmt(lvl)} tersentuh lebih dulu ({R.wib(rec['entry_t'], False)}). Batalkan pending order."
            if R.send(txt): rec["sent"].append("batal"); changed = True
        # pesan 3: hasil
        if rec.get("status") in ("SL", "CLOSE") and "hasil" not in rec["sent"]:
            done = [x for x in days.values() if x.get("status") in ("SL", "CLOSE") and not x["senin"]]
            tot = sum(x["r_net"] for x in done); win = sum(1 for x in done if x["r"] > 0)
            lab = "kena SL" if rec["status"] == "SL" else "ditutup akhir hari"
            txt = (f"<b>{tag} HASIL — {lab}</b>\nEntry {fmt(rec['entry'])} → {fmt(rec['exit'])}: <b>{rec['r']:+.2f}R</b> ({rec['r_net']:+.2f}R setelah fee)\n"
                   + (f"Forward test LW (Sel–Jum): {len(done)} transaksi · win {win/len(done)*100:.0f}% · total {tot:+.2f}R" if done else ""))
            if R.send(txt): rec["sent"].append("hasil"); changed = True
    if changed:
        R.save(STATE, {"days": sorted(days.values(), key=lambda x: x["day"])})
