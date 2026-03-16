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

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Permite que el HTML local llame al servidor

# ─── SERVIR EL FRONTEND ───────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

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
        rsi = calc_rsi(closes)
        ema50 = calc_ema(closes, 50)
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

        result = {
            'symbol':      symbol,
            'shortName':   short_name,
            'currency':    currency,
            'price':       price,
            'prevClose':   prev_close,
            'chgPct':      chg_pct,
            'rsi':         rsi,
            'ema50':       ema50,
            'ema200':      ema200,
            'sma50':       sma50,
            'sma200':      sma200,
            'distEma50':   dist_ema50,
            'distEma200':  dist_ema200,
            'mom3m':       mom3,
            'mom6m':       mom6,
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
