"""Radar XAU/USDT — pemindai setup v1 (breakout range Asia) + alert Telegram.

Dijalankan GitHub Actions tiap 15 menit saat sesi London/NY dan tiap jam di luar sesi.
Logika identik dengan backtest #1 (19 Okt 2026, lihat backtest/bt.py):
  - tren H4 : close > EMA50 > EMA200 (long) / kebalikannya (short), H4 terakhir yang sudah tutup sebelum candle H1 pemicu dibuka
  - tren D1 : EMA20 > EMA50 harian (candle UTC terakhir yang sudah tutup)
  - veto makro : skor DXY 5h + yield riil 10Y 5h, data bertanggal <= hari ini - 2
  - pemicu : H1 (mulai 07-15 UTC) tutup pertama kali di luar range Asia (00-07 UTC)
  - entry open H1 berikutnya, SL 2xATR14 H1, TP 2R, batas 24 jam, satu posisi per waktu
Hasil dicek di candle M15; kalau SL dan TP kena di candle yang sama dihitung SL.
"""
import json, os, sys, time, csv, io, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd

INST = "XAU-USDT-SWAP"
CT_VAL = 0.001          # 1 kontrak OKX = 0,001 XAU
FEE_RT = 0.0010         # fee taker 0,05% x 2
SL_ATR, TP_R, MAX_H = 2.0, 2.0, 24
RISK_USDT = float(os.environ.get("RISK_USDT", "10"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "signals.json")
MACRO = os.path.join(ROOT, "data", "macro.json")
WIB = timezone(timedelta(hours=7))
YAHOO_PROXY = "https://yahoo-proxy.indraanggriawan17.workers.dev/?url="
DRY = os.environ.get("DRY_RUN") == "1"
HARI = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# ------------------------------------------------------------------ util
def http_json(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            last = e; time.sleep(2 * (i + 1))
    raise RuntimeError(f"gagal ambil {url.split('?')[0]}: {last}")

def http_text(url, tries=4):
    # FRED menggantung untuk User-Agent browser/tak dikenal -> pakai UA curl
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
            return urllib.request.urlopen(req, timeout=30).read().decode()
        except Exception as e:
            last = e; time.sleep(2 * (i + 1))
    raise RuntimeError(f"gagal ambil {url.split('?')[0]}: {last}")

def wib(ts, with_day=True):
    t = pd.Timestamp(ts).tz_convert(WIB) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC").tz_convert(WIB)
    s = f"{t.day} {BULAN[t.month-1]} {t:%H:%M} WIB"
    return (HARI[t.dayofweek] + " " + s) if with_day else s

def f1(x): return f"{x:,.1f}"

# ------------------------------------------------------------------ data OKX
def okx_candles(bar, n):
    """n candle terakhir yang SUDAH tutup (confirm=1), urut naik. Index = waktu buka (UTC)."""
    rows, after = {}, None
    while len(rows) < n + 2:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={INST}&bar={bar}&limit=100" + (f"&after={after}" if after else "")
        d = http_json(url)
        if d.get("code") != "0": raise RuntimeError(f"OKX error {d.get('code')}: {d.get('msg')}")
        d = d["data"]
        if not d: break
        for r in d: rows[int(r[0])] = r
        after = d[-1][0]
        time.sleep(0.15)
    out = sorted(rows.values(), key=lambda r: int(r[0]))
    df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), r[8]] for r in out],
                      columns=["ts", "o", "h", "l", "c", "confirm"])
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    live = df[df.confirm != "1"]
    df = df[df.confirm == "1"].drop(columns=["ts", "confirm"])
    return df.tail(n), live

def okx_15m_since(start):
    rows, after = {}, None
    while True:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={INST}&bar=15m&limit=100" + (f"&after={after}" if after else "")
        d = http_json(url)["data"]
        if not d: break
        for r in d: rows[int(r[0])] = r
        after = d[-1][0]
        if int(after) <= int(start.timestamp() * 1000): break
        time.sleep(0.15)
    out = sorted(rows.values(), key=lambda r: int(r[0]))
    df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), r[8]] for r in out],
                      columns=["ts", "o", "h", "l", "c", "confirm"])
    df.index = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df[df.index >= start]

# ------------------------------------------------------------------ indikator (sama dengan backtest)
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def prep(h1, h4, d1):
    h4 = h4.copy(); d1 = d1.copy(); h1 = h1.copy()
    h4["ema50"] = ema(h4.c, 50); h4["ema200"] = ema(h4.c, 200)
    h4["trend"] = np.where((h4.c > h4.ema50) & (h4.ema50 > h4.ema200), 1,
                  np.where((h4.c < h4.ema50) & (h4.ema50 < h4.ema200), -1, 0))
    d1["ema20"] = ema(d1.c, 20); d1["ema50"] = ema(d1.c, 50)
    d1["dtrend"] = np.where(d1.ema20 > d1.ema50, 1, -1)
    h1["atr"] = atr(h1)
    return h1, h4, d1

# ------------------------------------------------------------------ makro
def yahoo_daily(sym):
    q = urllib.parse.quote(sym, safe="")
    urls = [f"https://query1.finance.yahoo.com/v8/finance/chart/{q}?interval=1d&range=6mo",
            f"https://query2.finance.yahoo.com/v8/finance/chart/{q}?interval=1d&range=6mo"]
    urls.append(YAHOO_PROXY + urllib.parse.quote(urls[0], safe=""))
    err = None
    for u in urls:
        try:
            r = http_json(u, tries=2)["chart"]["result"][0]
            s = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.to_datetime(r["timestamp"], unit="s", utc=True).normalize()).dropna()
            return s[~s.index.duplicated(keep="last")]
        except Exception as e:
            err = e
    raise RuntimeError(f"Yahoo {sym}: {err}")

def fred_dfii10():
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=200)).date().isoformat()
    txt = http_text(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&cosd={start}")
    rows = [r for r in csv.reader(io.StringIO(txt))][1:]
    s = pd.Series([pd.to_numeric(r[1], errors="coerce") for r in rows], index=pd.to_datetime([r[0] for r in rows], utc=True)).dropna()
    return s[s.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=200)]

def macro_table():
    dxy = yahoo_daily("DX-Y.NYB"); ry = fred_dfii10()
    m = pd.DataFrame({"dxy": dxy, "ry": ry}).sort_index().ffill()
    m["dxy_ch5"] = m.dxy.pct_change(5) * 100
    m["ry_ch5"] = m.ry.diff(5)
    m["s_dxy"] = np.where(m.dxy_ch5 < -0.5, 1, np.where(m.dxy_ch5 > 0.5, -1, 0))
    m["s_ry"] = np.where(m.ry_ch5 < -0.05, 1, np.where(m.ry_ch5 > 0.05, -1, 0))
    m["macro"] = m.s_dxy + m.s_ry
    m = m.dropna().tail(40)
    return [dict(date=str(i.date()), dxy=round(r.dxy, 3), ry=round(r.ry, 3), dxy_ch5=round(r.dxy_ch5, 3), ry_ch5=round(r.ry_ch5, 3),
                 s_dxy=int(r.s_dxy), s_ry=int(r.s_ry), macro=int(r.macro)) for i, r in m.iterrows()]

def macro_for(rows, day):
    """Skor makro yang berlaku untuk hari `day` (UTC): baris bertanggal <= day - 2 hari (sama dengan backtest)."""
    lim = (pd.Timestamp(day).normalize() - pd.Timedelta(days=2)).date().isoformat()
    ok = [r for r in rows if r["date"] <= lim]
    return ok[-1] if ok else None

# ------------------------------------------------------------------ evaluasi satu candle H1
def evaluate(t, h1, h4, d1, mrows):
    """t = waktu buka candle H1 yang sudah tutup. Return dict status (arah 0 kalau tidak ada setup)."""
    t = pd.Timestamp(t)
    bar = h1.loc[t]; prev = h1.loc[:t].iloc[-2]
    h4c = h4[h4.index + pd.Timedelta(hours=4) <= t]
    d1c = d1[d1.index + pd.Timedelta(days=1) <= t]
    day = h1[(h1.index.normalize() == t.normalize()) & (h1.index.hour < 7)]
    mr = macro_for(mrows, t)
    st = dict(t=t.isoformat(), close=bar.c, atr=float(bar.atr), trend=int(h4c.trend.iloc[-1]), dtrend=int(d1c.dtrend.iloc[-1]),
              h4_close=float(h4c.c.iloc[-1]), ema50=float(h4c.ema50.iloc[-1]), ema200=float(h4c.ema200.iloc[-1]),
              d_ema20=float(d1c.ema20.iloc[-1]), d_ema50=float(d1c.ema50.iloc[-1]),
              macro=(mr["macro"] if mr else 0), macro_row=mr,
              ah=float(day.h.max()) if len(day) == 7 else None, al=float(day.l.min()) if len(day) == 7 else None, dir=0)
    hr, dow = t.hour, t.dayofweek
    if dow >= 5 or (dow == 4 and hr >= 16) or not (7 <= hr <= 15) or st["ah"] is None: return st
    tr = st["trend"]
    if tr == 1 and bar.c > st["ah"] and prev.c <= st["ah"]: d = 1
    elif tr == -1 and bar.c < st["al"] and prev.c >= st["al"]: d = -1
    else: return st
    st["raw_dir"] = d
    if st["dtrend"] != d: st["blocked"] = "tren D1 berlawanan"; return st
    if np.sign(st["macro"]) * d < 0: st["blocked"] = "veto makro"; return st
    st["dir"] = d
    return st

# ------------------------------------------------------------------ pantau posisi
def resolve(sig, m15):
    """Cek apakah sinyal terbuka sudah kena SL/TP/batas waktu. Return dict update atau None."""
    e, d, risk = sig["entry"], sig["dir"], sig["risk"]
    sl, tp = sig["sl"], sig["tp"]
    entry_t = pd.Timestamp(sig["entry_t"])
    bars = m15[(m15.index >= entry_t)]
    bars = bars[bars.confirm == "1"].head(MAX_H * 4)
    for t, b in bars.iterrows():
        hit_sl = (b.l <= sl) if d == 1 else (b.h >= sl)
        hit_tp = (b.h >= tp) if d == 1 else (b.l <= tp)
        if hit_sl: return dict(status="SL", exit_t=(t + pd.Timedelta(minutes=15)).isoformat(), exit=sl, r=-1.0)
        if hit_tp: return dict(status="TP", exit_t=(t + pd.Timedelta(minutes=15)).isoformat(), exit=tp, r=TP_R)
    if len(bars) >= MAX_H * 4:
        last = bars.iloc[-1]
        return dict(status="WAKTU", exit_t=(bars.index[-1] + pd.Timedelta(minutes=15)).isoformat(), exit=float(last.c), r=d * (last.c - e) / risk)
    return None

# ------------------------------------------------------------------ telegram
def send(text):
    if DRY:
        print("---- TELEGRAM (dry run) ----\n" + text + "\n----------------------------"); return True
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diisi di Secrets; pesan tidak terkirim:\n" + text); return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode()
    for i in range(3):
        try:
            r = json.load(urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=30))
            if r.get("ok"): return True
        except urllib.error.HTTPError as e:
            print(f"::error::Telegram menolak pesan ({e.code}): {e.read().decode()[:200]}"); return False
        except Exception as e:
            print("telegram gagal:", e); time.sleep(3)
    return False

def tick(ok): return "✓" if ok else "✗"
def sgn(m): return f"{m:+d}" if m else "0"

def msg_setup(s):
    d = s["dir"]; arah = "LONG" if d == 1 else "SHORT"
    lvl = s["ah"] if d == 1 else s["al"]
    size = RISK_USDT / s["risk"]; ct = int(size / CT_VAL)
    mr = s.get("macro_row") or {}
    fee_r = FEE_RT * s["entry"] / s["risk"]
    return (f"<b>[XAU] SETUP v1 — {arah}</b>\n"
            f"Candle H1 {wib(s['t'], False)}–{(pd.Timestamp(s['t']) + pd.Timedelta(hours=1)).tz_convert(WIB):%H:%M} tutup <b>{f1(s['close'])}</b>, {'di atas high' if d == 1 else 'di bawah low'} Asia {f1(lvl)}\n\n"
            f"Entry acuan: <b>{f1(s['entry'])}</b> (open H1 {wib(s['entry_t'], False)})\n"
            f"SL: <b>{f1(s['sl'])}</b> (jarak {s['risk']:.1f})\n"
            f"TP: <b>{f1(s['tp'])}</b> (jarak {s['risk']*TP_R:.1f})\n"
            f"Batas waktu: {wib(s['deadline'])}\n\n"
            f"H4 {tick(s['trend']==d)} · D1 {tick(s['dtrend']==d)} · Makro {sgn(s['macro'])} {tick(s['macro']*d>=0)}"
            + (f" (DXY {mr['dxy_ch5']:+.2f}%, yield riil {mr['ry_ch5']:+.2f}, data {mr['date']})\n" if mr else " (data makro tidak tersedia, dianggap netral)\n") +
            f"Ukuran per {RISK_USDT:.0f} USDT risiko: {ct*CT_VAL:.3f} XAU = {ct} kontrak OKX · fee ≈ {fee_r:.2f}R\n\n"
            + ("⚠ <b>Rilis 3★ AS selama posisi:</b>\n" + "\n".join("• " + x for x in s["events"]) + "\n\n" if s.get("events") else "") +
            f"Level dari harga OKX. Di MetaTrader pakai jaraknya (SL {s['risk']:.1f} / TP {s['risk']*TP_R:.1f}).\n"
            f"<i>Masih forward test — catat hasilnya.</i>")

def msg_result(s, sigs):
    arah = "LONG" if s["dir"] == 1 else "SHORT"
    lab = {"TP": "kena TP", "SL": "kena SL", "WAKTU": "ditutup (24 jam habis)"}[s["status"]]
    done = [x for x in sigs if x.get("status") in ("TP", "SL", "WAKTU")]
    wins = sum(1 for x in done if x["r"] > 0); tot = sum(x["r_net"] for x in done)
    return (f"<b>[XAU] HASIL — {arah} {lab}</b>\n"
            f"Entry {f1(s['entry'])} → keluar {f1(s['exit'])} ({wib(s['exit_t'])})\n"
            f"Hasil: <b>{s['r']:+.2f}R</b> ({s['r_net']:+.2f}R setelah fee)\n\n"
            f"Forward test: {len(done)} sinyal selesai · win {wins/len(done)*100:.0f}% · total {tot:+.2f}R setelah fee\n"
            f"Pembanding backtest: win 55% · rata-rata +0,26R per sinyal")

# ------------------------------------------------------------------ main
def load_state():
    try: return json.load(open(STATE))
    except Exception: return {"signals": [], "last_bar": None, "last_error_sent": None}

def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = json.dumps(obj, ensure_ascii=False, indent=1, default=float)
    try:
        if open(path).read() == new: return
    except Exception: pass
    open(path, "w").write(new)

def run(now=None):
    state = load_state(); first_run = state.get("last_bar") is None
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    # makro: ambil ulang, kalau gagal pakai file terakhir
    try:
        old = json.load(open(MACRO))
    except Exception:
        old = None
    try:
        mrows = macro_table()
        save(MACRO, {"updated": now.isoformat(), "rows": mrows, "stale": False})
    except Exception as e:
        print("makro gagal, pakai data terakhir:", e)
        mrows = old["rows"] if old else []
        if old and not old.get("stale"): old["stale"] = True; save(MACRO, old)

    try:
        import kalender as K
        cal = K.refresh(state, now)
    except Exception as e:
        print("kalender error:", e); K = None; cal = {"events": []}

    h1, _ = okx_candles("1H", 200)
    h4, _ = okx_candles("4H", 1200)
    d1, _ = okx_candles("1Dutc", 300)
    h1, h4, d1 = prep(h1, h4, d1)
    sigs = state["signals"]

    # 1) pantau sinyal terbuka
    opn = [s for s in sigs if s.get("status") == "OPEN"]
    if opn:
        m15 = okx_15m_since(pd.Timestamp(opn[0]["entry_t"]))
        for s in opn:
            u = resolve(s, m15)
            if u:
                s.update(u); s["r_net"] = round(s["r"] - FEE_RT * s["entry"] / s["risk"], 3); s["r"] = round(s["r"], 3)
                send(msg_result(s, sigs))

    # 2) cek candle H1 yang baru tutup
    last = pd.Timestamp(state["last_bar"]) if state.get("last_bar") else h1.index[-1] - pd.Timedelta(hours=1)
    last = max(last, h1.index[-1] - pd.Timedelta(hours=6))   # jangan mengejar sinyal yang sudah lebih dari 6 jam
    for t in h1.index[h1.index > last]:
        st = evaluate(t, h1, h4, d1, mrows)
        state["last_bar"] = t.isoformat()
        if st["dir"] == 0:
            if st.get("blocked"): print(f"{t}: breakout {st['raw_dir']:+d} diblokir ({st['blocked']})")
            continue
        entry_t = t + pd.Timedelta(hours=1)
        busy = any(s.get("status") == "OPEN" or (s.get("exit_t") and pd.Timestamp(s["exit_t"]) > entry_t) for s in sigs)
        if busy: print(f"{t}: setup {st['dir']:+d} dilewati (masih ada posisi)"); continue
        m15 = okx_15m_since(entry_t)
        if not len(m15): print(f"{t}: candle entry belum ada"); state["last_bar"] = (t - pd.Timedelta(hours=1)).isoformat(); break
        e = float(m15.iloc[0].o); risk = SL_ATR * st["atr"]; d = st["dir"]
        s = dict(st, id=t.strftime("%Y%m%d%H"), entry_t=entry_t.isoformat(), entry=e, risk=round(risk, 3),
                 sl=round(e - d * risk, 2), tp=round(e + d * TP_R * risk, 2), deadline=(entry_t + pd.Timedelta(hours=MAX_H)).isoformat(),
                 status="OPEN", sent_at=now.isoformat())
        s.pop("macro_row", None); s["macro_detail"] = st.get("macro_row")
        if K:
            s["events"] = [K.fmt_event(x) for x in K.upcoming(cal, entry_t - pd.Timedelta(hours=1), entry_t + pd.Timedelta(hours=MAX_H))]
        sigs.append(s)
        send(msg_setup(dict(s, macro_row=st.get("macro_row"))))
        u = resolve(s, okx_15m_since(entry_t))   # kalau notifikasi telat dan sudah selesai
        if u:
            s.update(u); s["r_net"] = round(s["r"] - FEE_RT * e / risk, 3); s["r"] = round(s["r"], 3); send(msg_result(s, sigs))

    if not state.get("hello_sent"):
        st = evaluate(h1.index[-1], h1, h4, d1, mrows)
        tr = {1: "naik", -1: "turun", 0: "netral"}
        state["hello_sent"] = send(f"<b>[XAU] Radar XAU/USDT aktif</b>\nAlert setup v1 dan hasilnya akan dikirim ke chat ini.\n"
             f"Sekarang: H4 {tr[st['trend']]}, D1 {tr[st['dtrend']]}, makro {sgn(st['macro'])}.")
    if K:
        try: K.process(cal, now)
        except Exception as e: print("proses rilis error:", e)
    state["updated"] = now.isoformat()
    save(STATE, state)

if __name__ == "__main__":
    if not DRY and not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")):
        print("::error::Secret TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID kosong. Isi di Settings > Secrets and variables > Actions > tab Secrets > Repository secrets.")
        try: run()
        finally: sys.exit(1)
    try:
        run()
    except Exception as e:
        print("ERROR:", e)
        st = load_state(); now = datetime.now(timezone.utc)
        lastsent = st.get("last_error_sent")
        if not lastsent or (now - datetime.fromisoformat(lastsent)).total_seconds() > 6 * 3600:
            if send(f"<b>[XAU] Radar gagal jalan</b>\n{str(e)[:300]}\nSinyal bisa terlewat sampai ini beres."):
                st["last_error_sent"] = now.isoformat(); save(STATE, st)
        sys.exit(1)
