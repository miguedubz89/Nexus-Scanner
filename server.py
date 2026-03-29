"""
NEXUS SCANNER — Servidor Proxy Local
=====================================
Instalación (una sola vez):
    pip install yfinance flask flask-cors pandas numpy

Uso:
    python server.py

Luego abrí market-scanner.html en tu browser.
El servidor corre en http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Permite que el HTML local llame al servidor

# ─── INDICADORES TÉCNICOS ─────────────────────────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calc_ema(closes, period):
    if len(closes) < period:
        return None
    s = pd.Series(closes)
    return round(float(s.ewm(span=period, adjust=False).mean().iloc[-1]), 4)

def calc_sma(closes, period):
    if len(closes) < period:
        return None
    return round(float(np.mean(closes[-period:])), 4)

def pct_dist(price, ref):
    if ref is None or ref == 0:
        return None
    return round((price - ref) / ref * 100, 2)

def momentum(closes, days):
    if len(closes) < days + 1:
        return None
    return round((closes[-1] / closes[-days] - 1) * 100, 2)

def calc_adx(hist, period=14):
    """
    Average Directional Index (ADX) — Wilder's method.
    """
    try:
        if len(hist) < period * 2 + 1:
            return None

        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)
        close = hist['Close'].values.astype(float)

        n = len(close)
        tr  = np.zeros(n)
        pdm = np.zeros(n)
        ndm = np.zeros(n)

        for i in range(1, n):
            hl  = high[i]  - low[i]
            hpc = abs(high[i]  - close[i-1])
            lpc = abs(low[i]   - close[i-1])
            tr[i] = max(hl, hpc, lpc)

            up   = high[i]  - high[i-1]
            down = low[i-1] - low[i]
            pdm[i] = up   if (up > down and up > 0)   else 0.0
            ndm[i] = down if (down > up and down > 0) else 0.0

        atr  = np.zeros(n)
        apdm = np.zeros(n)
        andm = np.zeros(n)

        atr[period]  = np.sum(tr[1:period+1])
        apdm[period] = np.sum(pdm[1:period+1])
        andm[period] = np.sum(ndm[1:period+1])

        for i in range(period+1, n):
            atr[i]  = atr[i-1]  - atr[i-1]/period  + tr[i]
            apdm[i] = apdm[i-1] - apdm[i-1]/period + pdm[i]
            andm[i] = andm[i-1] - andm[i-1]/period + ndm[i]

        pdi = np.where(atr > 0, 100 * apdm / atr, 0.0)
        ndi = np.where(atr > 0, 100 * andm / atr, 0.0)
        dx  = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)

        adx_arr = np.zeros(n)
        adx_arr[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, n):
            adx_arr[i] = (adx_arr[i-1] * (period-1) + dx[i]) / period

        result = adx_arr[-1]
        return round(float(result), 2) if result > 0 else None

    except Exception:
        return None


def calc_squeeze_momentum(hist, length_bb=20, mult_bb=2.0, length_kc=20, mult_kc=1.5):
    """
    Squeeze Momentum Indicator — LazyBear (TTM Squeeze)
    =====================================================
    Lógica:
      - Squeeze ON  (sqzOn=True)  → BB está DENTRO de KC  → mercado comprimido, energía acumulándose
      - Squeeze OFF (sqzOff=True) → BB acaba de salir de KC → energía liberada, posible ruptura
      - No Squeeze  (ambos False) → BB está fuera de KC normalmente

    Momentum (sqzMom):
      - Oscilador = linreg( close - avg(avg(highest_high, lowest_low), SMA(close, KC_period)) , length, 0 )
      - Positivo y subiendo  → momentum alcista creciente   (verde brillante en TV)
      - Positivo y bajando   → momentum alcista debilitándose (verde oscuro)
      - Negativo y bajando   → momentum bajista creciente   (rojo brillante)
      - Negativo y subiendo  → momentum bajista debilitándose (rojo oscuro)

    Retorna dict con: sqzOn, sqzOff, sqzMom, sqzMomPrev
    """
    try:
        if len(hist) < length_bb + 10:
            return None

        close = hist['Close'].values.astype(float)
        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)
        n     = len(close)

        # ── Bollinger Bands (SMA + stddev) ──────────────
        bb_sma    = np.array([np.mean(close[max(0,i-length_bb+1):i+1]) for i in range(n)])
        bb_std    = np.array([np.std( close[max(0,i-length_bb+1):i+1], ddof=0) for i in range(n)])
        bb_upper  = bb_sma + mult_bb * bb_std
        bb_lower  = bb_sma - mult_bb * bb_std

        # ── Keltner Channel (EMA + ATR) ──────────────────
        # True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr[0] = high[0] - low[0]

        # EMA del TR = ATR
        kc_ma  = pd.Series(close).ewm(span=length_kc, adjust=False).mean().values
        kc_atr = pd.Series(tr).ewm(span=length_kc, adjust=False).mean().values
        kc_upper = kc_ma + mult_kc * kc_atr
        kc_lower = kc_ma - mult_kc * kc_atr

        # ── Squeeze detection ────────────────────────────
        # sqzOn  = BB dentro de KC
        sqz_on_arr  = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        # sqzOff = BB acaba de salir (antes estaba dentro, ahora no)
        sqz_off_arr = np.zeros(n, dtype=bool)
        for i in range(1, n):
            sqz_off_arr[i] = sqz_on_arr[i-1] and not sqz_on_arr[i]

        # ── Momentum value (LazyBear) ────────────────────
        # val = close - mean( mean(highest(high,kc_len), lowest(low,kc_len)), sma(close,kc_len) )
        val = np.zeros(n)
        for i in range(length_kc - 1, n):
            hh = np.max(high[i-length_kc+1:i+1])
            ll = np.min(low[ i-length_kc+1:i+1])
            sma_c = np.mean(close[i-length_kc+1:i+1])
            val[i] = close[i] - (((hh + ll) / 2) + sma_c) / 2

        # Linear regression of val over length_bb bars (= linreg(val, length, 0))
        sqz_mom = np.zeros(n)
        for i in range(length_bb - 1, n):
            y = val[i-length_bb+1:i+1]
            x = np.arange(length_bb, dtype=float)
            # simple linreg, take last point (x = length_bb - 1)
            xm = np.mean(x); ym = np.mean(y)
            b  = np.sum((x - xm) * (y - ym)) / (np.sum((x - xm)**2) + 1e-10)
            a  = ym - b * xm
            sqz_mom[i] = a + b * (length_bb - 1)

        # ── Resultado final (último valor) ───────────────
        sqz_on_val   = bool(sqz_on_arr[-1])
        sqz_off_val  = bool(sqz_off_arr[-1])
        mom_val      = round(float(sqz_mom[-1]), 4)
        mom_prev_val = round(float(sqz_mom[-2]), 4) if n >= 2 else mom_val

        return {
            'sqzOn':      sqz_on_val,
            'sqzOff':     sqz_off_val,
            'sqzMom':     mom_val,
            'sqzMomPrev': mom_prev_val,
        }

    except Exception:
        traceback.print_exc()
        return None


# ─── ENDPOINT PRINCIPAL ───────────────────────────────────────────────────────

@app.route('/quote', methods=['GET'])
def get_quote():
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400

    try:
        ticker = yf.Ticker(symbol)

        # Historial 1 año (necesario para EMA200, momentums)
        hist = ticker.history(period='1y', interval='1d', auto_adjust=True)

        if hist.empty or len(hist) < 5:
            return jsonify({'error': f'No data for {symbol}'}), 404

        closes  = hist['Close'].dropna().values.tolist()
        volumes = hist['Volume'].dropna().values.tolist()

        price      = round(closes[-1], 4)
        prev_close = round(closes[-2], 4) if len(closes) >= 2 else price
        chg_pct    = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0

        # Indicadores técnicos
        rsi         = calc_rsi(closes)
        adx         = calc_adx(hist)
        ema50       = calc_ema(closes, 50)
        ema200      = calc_ema(closes, 200)
        sma50       = calc_sma(closes, 50)
        sma200      = calc_sma(closes, 200)
        dist_ema50  = pct_dist(price, ema50)
        dist_ema200 = pct_dist(price, ema200)
        mom3        = momentum(closes, 63)    # ~3 meses
        mom6        = momentum(closes, 126)   # ~6 meses
        mom12       = momentum(closes, 252)   # ~12 meses (1 año)

        # ── Squeeze Momentum (LazyBear) ──────────────────
        squeeze     = calc_squeeze_momentum(hist)

        # Volúmenes
        vol_today = int(volumes[-1]) if volumes else 0
        avg_vol20 = int(np.mean(volumes[-20:])) if len(volumes) >= 20 else vol_today

        # 52w high
        high52         = round(float(max(closes)), 4)
        dist_from_high = pct_dist(price, high52)

        # Sparkline últimos 20 cierres
        spark = [round(c, 2) for c in closes[-20:]]

        # Fundamentals
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        def safe(key, default=None):
            v = info.get(key, default)
            return v if v not in (None, 'None', '', 'N/A') else default

        market_cap = safe('marketCap')
        pe_ratio   = safe('trailingPE') or safe('forwardPE')
        beta       = safe('beta')
        div_yield  = safe('dividendYield')
        if div_yield is not None:
            div_yield = round(div_yield * 100, 4)

        currency   = safe('currency', 'USD')
        short_name = safe('shortName', symbol)

        result = {
            'symbol':      symbol,
            'shortName':   short_name,
            'currency':    currency,
            'price':       price,
            'prevClose':   prev_close,
            'chgPct':      chg_pct,
            'rsi':         rsi,
            'adx':         adx,
            'ema50':       ema50,
            'ema200':      ema200,
            'sma50':       sma50,
            'sma200':      sma200,
            'distEma50':   dist_ema50,
            'distEma200':  dist_ema200,
            'mom3m':       mom3,
            'mom6m':       mom6,
            'mom12m':      mom12,           # ← NUEVO: momentum anual
            # ── Squeeze Momentum LazyBear ──────────────
            'sqzOn':       squeeze['sqzOn']      if squeeze else None,
            'sqzOff':      squeeze['sqzOff']     if squeeze else None,
            'sqzMom':      squeeze['sqzMom']     if squeeze else None,
            'sqzMomPrev':  squeeze['sqzMomPrev'] if squeeze else None,
            # ───────────────────────────────────────────
            'volume':      vol_today,
            'avgVol20':    avg_vol20,
            'high52':      high52,
            'distFromHigh':dist_from_high,
            'marketCap':   market_cap,
            'pe':          round(pe_ratio, 2) if pe_ratio else None,
            'beta':        round(beta, 3) if beta else None,
            'divYield':    div_yield,
            'spark':       spark,
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/quotes', methods=['POST'])
def get_quotes():
    """Recibe lista de symbols y devuelve todos en paralelo."""
    data    = request.get_json()
    symbols = data.get('symbols', [])
    if not symbols:
        return jsonify([])

    results = []
    for sym in symbols:
        try:
            r = app.test_client().get(f'/quote?symbol={sym}')
            import json
            results.append(json.loads(r.data))
        except Exception as e:
            results.append({'symbol': sym, 'error': str(e)})

    return jsonify(results)


@app.route('/precio', methods=['GET'])
def get_precio():
    """Fetch rápido: solo precio actual, cambio% y volumen del día."""
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period='5d', interval='1d', auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return jsonify({'error': 'No data'}), 404
        closes  = hist['Close'].dropna().values.tolist()
        volumes = hist['Volume'].dropna().values.tolist()
        price      = round(closes[-1], 4)
        prev_close = round(closes[-2], 4)
        chg_pct    = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0
        volume     = int(volumes[-1]) if volumes else 0
        return jsonify({
            'symbol':    symbol,
            'price':     price,
            'prevClose': prev_close,
            'chgPct':    chg_pct,
            'volume':    volume,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'NEXUS SCANNER proxy running'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
