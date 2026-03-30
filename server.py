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

app = Flask(__name__)
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
    Requiere columnas High, Low, Close en el DataFrame.
    Interpretación:
        < 20  → sin tendencia clara
        20-25 → tendencia débil
        25-50 → tendencia fuerte
        > 50  → tendencia muy fuerte
    """
    try:
        if len(hist) < period * 2 + 1:
            return None

        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)
        close = hist['Close'].values.astype(float)

        n = len(close)

        # True Range
        tr  = np.zeros(n)
        pdm = np.zeros(n)   # +DM
        ndm = np.zeros(n)   # -DM

        for i in range(1, n):
            hl  = high[i]  - low[i]
            hpc = abs(high[i]  - close[i-1])
            lpc = abs(low[i]   - close[i-1])
            tr[i] = max(hl, hpc, lpc)

            up   = high[i]  - high[i-1]
            down = low[i-1] - low[i]
            pdm[i] = up   if (up > down and up > 0)   else 0.0
            ndm[i] = down if (down > up and down > 0) else 0.0

        # Wilder smoothing (suma inicial + RMA)
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

        # +DI / -DI
        pdi = np.where(atr > 0, 100 * apdm / atr, 0.0)
        ndi = np.where(atr > 0, 100 * andm / atr, 0.0)

        # DX
        dx = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)

        # ADX = Wilder smooth de DX
        adx_arr = np.zeros(n)
        adx_arr[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, n):
            adx_arr[i] = (adx_arr[i-1] * (period-1) + dx[i]) / period

        result = adx_arr[-1]
        return round(float(result), 2) if result > 0 else None

    except Exception:
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

        closes = hist['Close'].dropna().values.tolist()
        volumes = hist['Volume'].dropna().values.tolist()

        price = round(closes[-1], 4)
        prev_close = round(closes[-2], 4) if len(closes) >= 2 else price
        chg_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0

        # Indicadores técnicos
        rsi    = calc_rsi(closes)
        adx    = calc_adx(hist)   # ADX usa High/Low/Close del DataFrame completo
        ema50  = calc_ema(closes, 50)
        ema200 = calc_ema(closes, 200)
        sma50 = calc_sma(closes, 50)
        sma200 = calc_sma(closes, 200)
        dist_ema50 = pct_dist(price, ema50)
        dist_ema200 = pct_dist(price, ema200)
        mom3 = momentum(closes, 63)   # ~3 meses
        mom6 = momentum(closes, 126)  # ~6 meses

        # Volúmenes
        vol_today = int(volumes[-1]) if volumes else 0
        avg_vol20 = int(np.mean(volumes[-20:])) if len(volumes) >= 20 else vol_today

        # 52w high
        high52 = round(float(max(closes)), 4)
        dist_from_high = pct_dist(price, high52)

        # Sparkline últimos 20 cierres (redondeados)
        spark = [round(c, 2) for c in closes[-20:]]

        # Fundamentals via info (puede fallar en algunos tickers)
        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        def safe(key, default=None):
            v = info.get(key, default)
            return v if v not in (None, 'None', '', 'N/A') else default

        market_cap  = safe('marketCap')
        pe_ratio    = safe('trailingPE') or safe('forwardPE')
        beta        = safe('beta')
        div_yield   = safe('dividendYield')
        if div_yield is not None:
            div_yield = round(div_yield * 100, 4)

        currency    = safe('currency', 'USD')
        short_name  = safe('shortName', symbol)

        # ── CANSLIM FUNDAMENTALS ──────────────────────────────────────────────
        # C — Current quarterly earnings growth (YoY)
        eps_growth_q = None
        try:
            qf = ticker.quarterly_financials
            if qf is not None and not qf.empty and 'Net Income' in qf.index:
                ni = qf.loc['Net Income'].dropna()
                if len(ni) >= 5:
                    # Crecimiento YoY del trimestre más reciente vs mismo trimestre año anterior
                    eps_growth_q = round((ni.iloc[0] / ni.iloc[4] - 1) * 100, 1) if ni.iloc[4] != 0 else None
        except Exception:
            pass

        # A — Annual earnings growth (últimos 2 años)
        eps_growth_a = None
        try:
            af = ticker.financials  # anual
            if af is not None and not af.empty and 'Net Income' in af.index:
                ni_a = af.loc['Net Income'].dropna()
                if len(ni_a) >= 2:
                    eps_growth_a = round((ni_a.iloc[0] / ni_a.iloc[1] - 1) * 100, 1) if ni_a.iloc[1] != 0 else None
        except Exception:
            pass

        # ROE — Return on Equity (para L de CANSLIM)
        roe = None
        try:
            roe_raw = safe('returnOnEquity')
            if roe_raw is not None:
                roe = round(float(roe_raw) * 100, 1)
        except Exception:
            pass

        # Revenue growth trimestral YoY
        rev_growth_q = None
        try:
            qf2 = ticker.quarterly_financials
            if qf2 is not None and not qf2.empty and 'Total Revenue' in qf2.index:
                rv = qf2.loc['Total Revenue'].dropna()
                if len(rv) >= 5:
                    rev_growth_q = round((rv.iloc[0] / rv.iloc[4] - 1) * 100, 1) if rv.iloc[4] != 0 else None
        except Exception:
            pass

        # EPS aceleración: trimestre más reciente vs trimestre anterior (ambos YoY)
        eps_accel = None
        try:
            qf3 = ticker.quarterly_financials
            if qf3 is not None and not qf3.empty and 'Net Income' in qf3.index:
                ni3 = qf3.loc['Net Income'].dropna()
                if len(ni3) >= 6:
                    g_recent = (ni3.iloc[0] / ni3.iloc[4] - 1) * 100 if ni3.iloc[4] != 0 else None
                    g_prev   = (ni3.iloc[1] / ni3.iloc[5] - 1) * 100 if ni3.iloc[5] != 0 else None
                    if g_recent is not None and g_prev is not None:
                        eps_accel = round(g_recent - g_prev, 1)  # positivo = aceleración
        except Exception:
            pass

        # Profit margin
        profit_margin = None
        try:
            pm = safe('profitMargins')
            if pm is not None:
                profit_margin = round(float(pm) * 100, 1)
        except Exception:
            pass

        # Mom 12M (calculado con historial, más preciso que solo 252 días)
        mom12 = momentum(closes, 252) if len(closes) >= 253 else mom6

        result = {
            'symbol':        symbol,
            'shortName':     short_name,
            'currency':      currency,
            'price':         price,
            'prevClose':     prev_close,
            'chgPct':        chg_pct,
            'rsi':           rsi,
            'adx':           adx,
            'ema50':         ema50,
            'ema200':        ema200,
            'sma50':         sma50,
            'sma200':        sma200,
            'distEma50':     dist_ema50,
            'distEma200':    dist_ema200,
            'mom3m':         mom3,
            'mom6m':         mom6,
            'mom12m':        mom12,
            'volume':        vol_today,
            'avgVol20':      avg_vol20,
            'high52':        high52,
            'distFromHigh':  dist_from_high,
            'marketCap':     market_cap,
            'pe':            round(pe_ratio, 2) if pe_ratio else None,
            'beta':          round(beta, 3) if beta else None,
            'divYield':      div_yield,
            'spark':         spark,
            # CANSLIM fundamentals
            'epsGrowthQ':    eps_growth_q,    # C: crecimiento EPS trimestral YoY (%)
            'epsGrowthA':    eps_growth_a,    # A: crecimiento EPS anual YoY (%)
            'epsAccel':      eps_accel,       # aceleración earnings (trimestre reciente vs anterior)
            'revGrowthQ':    rev_growth_q,    # crecimiento Revenue trimestral YoY (%)
            'roe':           roe,             # L: Return on Equity (%)
            'profitMargin':  profit_margin,   # margen neto (%)
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/quotes', methods=['POST'])
def get_quotes():
    """Recibe lista de symbols y devuelve todos en paralelo."""
    data = request.get_json()
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
    """Fetch rápido: solo precio actual, cambio% y volumen del día.
    No descarga historial de 1 año — usado por el auto-refresh."""
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400
    try:
        ticker = yf.Ticker(symbol)
        # period=5d es suficiente para precio actual y prevClose
        hist = ticker.history(period='5d', interval='1d', auto_adjust=True)
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'NEXUS SCANNER proxy running'})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  NEXUS SCANNER — Proxy Server")
    print("="*50)
    print("  Corriendo en: http://localhost:5000")
    print("  Abrí market-scanner.html en tu browser")
    print("  Ctrl+C para detener")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
