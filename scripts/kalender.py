"""Kalender ekonomi AS berdampak tinggi (3 bintang) untuk Radar XAU/USDT.

- Jadwal + proyeksi + data sebelumnya: feed kalender ForexFactory (minggu berjalan), diambil maks tiap 3 jam.
- Angka aktual: dihitung dari FRED (pakai FRED_API_KEY) dengan membandingkan vintage hari rilis vs hari sebelumnya,
  jadi yang dipakai benar-benar angka rilis pertama, bukan revisi belakangan.
- Reaksi emas: gerak XAU-USDT-SWAP 15 menit, 1 jam, 4 jam setelah rilis (OKX M15).
Semua dicatat di data/events.json supaya nanti bisa diuji: apakah aktual vs proyeksi memprediksi arah emas.
Kesimpulan arah di pesan Telegram adalah pola umum, BELUM teruji di radar ini.
"""
import json, os, re
import numpy as np, pandas as pd
import radar_xau as R

CAL = os.path.join(R.ROOT, "data", "calendar.json")
EVT = os.path.join(R.ROOT, "data", "events.json")
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# judul ForexFactory -> (seri FRED, cara hitung, desimal, satuan). Urutan penting: yang spesifik dulu.
MAP = [
    ("Core CPI m/m", "CPILFESL", "mom", 1, "%"), ("CPI m/m", "CPIAUCSL", "mom", 1, "%"), ("CPI y/y", "CPIAUCSL", "yoy", 1, "%"),
    ("Core PCE Price Index m/m", "PCEPILFE", "mom", 1, "%"), ("Core PCE Price Index y/y", "PCEPILFE", "yoy", 1, "%"),
    ("Non-Farm Employment Change", "PAYEMS", "diff", 0, "K"), ("Unemployment Rate", "UNRATE", "level", 1, "%"),
    ("Average Hourly Earnings m/m", "CES0500000003", "mom", 1, "%"),
    ("GDP q/q", "A191RL1Q225SBEA", "level", 1, "%"),
    ("Core PPI m/m", "PPIFES", "mom", 1, "%"), ("PPI m/m", "PPIFIS", "mom", 1, "%"),
    ("Core Retail Sales m/m", "RSFSXMV", "mom", 1, "%"), ("Retail Sales m/m", "RSAFS", "mom", 1, "%"),
    ("Federal Funds Rate", "DFEDTARU", "level", 2, "%"),
]
INVERSE = ("Unemployment Rate",)   # angka lebih tinggi = ekonomi lebih lemah

def load(path, default):
    try: return json.load(open(path))
    except Exception: return default

def num(s):
    if s in (None, ""): return None
    m = re.match(r"^\s*(-?[\d.]+)\s*([KMB%]?)", str(s))
    return float(m.group(1)) if m else None

def mapping(title):
    for k, sid, how, dec, unit in MAP:
        if k in title: return sid, how, dec, unit
    return None

# ------------------------------------------------------------------ jadwal
def refresh(state, now):
    """Ambil feed ForexFactory maks tiap 3 jam; gabungkan dengan kalender tersimpan (simpan 21 hari terakhir)."""
    cal = load(CAL, {"events": []})
    last = state.get("calendar_checked")
    if last and (now - pd.Timestamp(last)).total_seconds() < 3 * 3600: return cal
    try:
        feed = R.http_json(FF_URL, tries=2)
    except Exception as e:
        print("kalender gagal diambil:", e); return cal
    state["calendar_checked"] = now.isoformat()
    ev = {(e["t"], e["title"]): e for e in cal["events"]}
    for x in feed:
        if x.get("country") != "USD" or x.get("impact") != "High": continue
        t = pd.Timestamp(x["date"]).tz_convert("UTC").isoformat()
        ev[(t, x["title"])] = dict(t=t, title=x["title"], forecast=x.get("forecast", ""), previous=x.get("previous", ""))
    keep = [e for e in ev.values() if pd.Timestamp(e["t"]) >= now - pd.Timedelta(days=21)]
    cal = {"events": sorted(keep, key=lambda e: (e["t"], e["title"]))}
    R.save(CAL, cal)
    return cal

def upcoming(cal, start, end):
    return [e for e in cal.get("events", []) if start <= pd.Timestamp(e["t"]) <= end]

def fmt_event(e, with_day=True):
    s = f"{e['title']} {R.wib(e['t'], with_day)}"
    bits = []
    if e.get("forecast"): bits.append(f"proyeksi {e['forecast']}")
    if e.get("previous"): bits.append(f"sebelumnya {e['previous']}")
    return s + (f" ({', '.join(bits)})" if bits else "")

# ------------------------------------------------------------------ aktual dari FRED
def fred_vintage(sid, day):
    key = os.environ.get("FRED_API_KEY")
    if not key: return None
    start = (pd.Timestamp(day) - pd.Timedelta(days=500)).date().isoformat()
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json"
           f"&realtime_start={day}&realtime_end={day}&observation_start={start}")
    obs = R.http_json(url, tries=2)["observations"]
    s = pd.Series({o["date"]: pd.to_numeric(o["value"], errors="coerce") for o in obs}).dropna()
    return s

def actual_for(e):
    m = mapping(e["title"])
    if not m: return None
    sid, how, dec, unit = m
    day = pd.Timestamp(e["t"]).tz_convert("America/New_York").date()
    if sid == "DFEDTARU":   # seri harian: batas atas suku bunga berlaku mulai hari setelah keputusan
        v = fred_vintage(sid, pd.Timestamp.now(tz="UTC").date().isoformat())
        if v is None: return None
        v = v[v.index > day.isoformat()]
        if v.empty: return None
        val = round(float(v.iloc[0]), dec)
        return dict(value=val, text=f"{val:.{dec}f}%", obs=str(v.index[0]), series=sid)
    now_v = fred_vintage(sid, day.isoformat())
    prev_v = fred_vintage(sid, (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat())
    if now_v is None or now_v.empty: return None
    # sudah dirilis kalau ada observasi baru, atau observasi terakhir berubah (revisi GDP, suku bunga)
    new = prev_v is None or prev_v.empty or now_v.index[-1] > prev_v.index[-1] or abs(now_v.iloc[-1] - prev_v.get(now_v.index[-1], np.nan)) > 1e-9
    if not new: return None
    v = now_v
    if how == "mom": val = (v.iloc[-1] / v.iloc[-2] - 1) * 100
    elif how == "yoy": val = (v.iloc[-1] / v.iloc[-13] - 1) * 100
    elif how == "diff": val = v.iloc[-1] - v.iloc[-2]
    else: val = v.iloc[-1]
    val = round(float(val), dec)
    return dict(value=val, text=(f"{val:.{dec}f}{unit}" if unit != "K" else f"{val:.0f}K"), obs=str(v.index[-1]), series=sid)

def verdict(e, act):
    f = num(e.get("forecast"))
    if act is None or f is None: return None, None
    diff = act["value"] - f
    if abs(diff) < 1e-9: return 0, "sesuai proyeksi → reaksi biasanya kecil"
    hot = diff > 0
    if any(k in e["title"] for k in INVERSE): hot = not hot
    if hot: return -1, "lebih panas/kuat dari proyeksi → biasanya dolar & yield naik, condong NEGATIF untuk emas"
    return 1, "lebih lemah/dingin dari proyeksi → biasanya dolar & yield turun, condong POSITIF untuk emas"

# ------------------------------------------------------------------ reaksi emas & pesan
def reaction(t):
    t = pd.Timestamp(t)
    try: m15 = R.okx_15m_since(t)
    except Exception: return {}
    m15 = m15[m15.confirm == "1"]
    if t not in m15.index: return {}
    o = float(m15.at[t, "o"]); out = {"open": o}
    for lab, mins in (("m15", 15), ("m60", 60), ("m240", 240)):
        end = t + pd.Timedelta(minutes=mins - 15)
        if end in m15.index: out[lab] = round((float(m15.at[end, "c"]) / o - 1) * 100, 3)
    return out

def process(cal, now):
    """Isi aktual & reaksi untuk rilis yang sudah lewat (maks 2 hari), kirim satu pesan per jam rilis."""
    log = load(EVT, {"events": []})
    rec = {(e["t"], e["title"]): e for e in log["events"]}
    changed = False
    for e in cal.get("events", []):
        t = pd.Timestamp(e["t"])
        if not (t + pd.Timedelta(minutes=15) <= now <= t + pd.Timedelta(days=2)): continue
        r = rec.setdefault((e["t"], e["title"]), dict(e, sent=False))
        r["forecast"], r["previous"] = e.get("forecast", ""), e.get("previous", "")
        if mapping(e["title"]) and "actual" not in r:
            try:
                a = actual_for(e)
                if a: r["actual"] = a; changed = True
            except Exception as ex:
                print("aktual gagal:", e["title"], ex)
        if "m240" not in r.get("reaction", {}):
            rx = reaction(e["t"])
            if rx and rx != r.get("reaction"): r["reaction"] = rx; changed = True
        if "actual" in r:
            d, why = verdict(r, r["actual"]); r["surprise_dir"], r["surprise_text"] = d, why
    # pesan: kelompokkan per jam rilis; kirim kalau semua yang bisa dipetakan sudah ada aktualnya (atau sudah lewat 3 jam)
    by_t = {}
    for r in rec.values(): by_t.setdefault(r["t"], []).append(r)
    for t, rs in sorted(by_t.items()):
        if all(x.get("sent") for x in rs): continue
        tt = pd.Timestamp(t)
        if now > tt + pd.Timedelta(days=2): continue
        need = [x for x in rs if mapping(x["title"])]
        ready = (all("actual" in x for x in need) and need) or now >= tt + pd.Timedelta(hours=3)
        if not ready or "m15" not in rs[0].get("reaction", {}): continue
        if not any(x.get("forecast") or mapping(x["title"]) for x in rs):   # mis. konferensi pers: cukup dicatat
            for x in rs: x["sent"] = True
            changed = True; continue
        lines = [f"<b>[XAU] RILIS 3★ AS — {R.wib(t)}</b>"]
        for x in rs:
            if x["title"] in ("FOMC Statement", "FOMC Press Conference") and len(rs) > 1: continue
            a = x.get("actual", {}).get("text") or "–"
            lines.append(f"\n<b>{x['title']}</b>: aktual {a} · proyeksi {x.get('forecast') or '–'} · sebelumnya {x.get('previous') or '–'}")
            if x.get("surprise_text"): lines.append("→ " + x["surprise_text"])
            elif mapping(x["title"]) and "actual" not in x: lines.append("→ angka aktual belum tersedia di FRED")
        rx = rs[0]["reaction"]
        mv = " · ".join(f"{lab} {rx[k]:+.2f}%" for k, lab in (("m15", "15 mnt"), ("m60", "1 jam")) if k in rx)
        lines.append(f"\nEmas setelah rilis: {mv}")
        lines.append("<i>Kesimpulan arah = pola umum, belum teruji di radar ini. Dicatat untuk evaluasi.</i>")
        if R.send("\n".join(lines)):
            for x in rs: x["sent"] = True
            changed = True
    if changed:
        keep = sorted(rec.values(), key=lambda e: (e["t"], e["title"]))
        R.save(EVT, {"events": keep})
