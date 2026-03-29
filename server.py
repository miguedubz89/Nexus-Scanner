"""
NEXUS SCANNER — Servidor Proxy Local
=====================================
Instalación (una sola vez):
    pip install yfinance flask flask-cors pandas numpy

Uso:
    python server.py

Luego abrí market-scanner.html en tu browser.
El servidor corre en http://localhost:5000

ACCIONES ARGENTINAS SIN ADR:
  - Si el símbolo termina en .BA  → se usa tal cual (ej: GGAL.BA)
  - Si el símbolo está en MERVAL_SET → se agrega .BA para Yahoo Finance
    Ejemplo: el usuario agrega "MOLI" → el server busca "MOLI.BA" en YF
             y devuelve el símbolo como "MOLI" al frontend.
  - Esto resuelve que MOLI, MORI, BYMA aparezcan en la sección Merval.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# ─── TICKERS ARGENTINOS SIN ADR (se buscan con .BA en Yahoo Finance) ──────────
MERVAL_SET = {
    'GGAL','YPFD','YPF','PAMP','BMA','SUPV','CEPU','EDN',
    'LOMA','COME','BYMA','ALUA','BHIP','BOLT','BPAT','BRIO',
    'MOLI','MORI','IRSA','LEDE','LONG','METR','MIRG','MOLA',
    'OEST','PATA','PECO','PERD','POLL','RIGO','ROSE','SAMI',
    'SEMI','TECO2','TGNO4','TGSU2','TRAN','TXAR','VALO',
    'APBR','HARG','INTR','INVJ','AGRO','CRES','RICH','HOLS',
    'CARC','CECO2','CELU','CGPA2','CTIO','CVH','DGCE','DGCU2',
    'DOME','DYCA','FERR','FIPL','GAMI','GCDI','GCLA','GRIM',
    'HAVA','HEBA','CADO','CAPX',
}

def resolve_symbol(sym: str):
    """
    Devuelve (yf_symbol, is_ar):
      - yf_symbol: el ticker real para Yahoo Finance
      - is_ar: True si es una acción argentina (cotiza en ARS)

    Reglas:
      - Termina en .BA           → argentino, usar tal cual
      - Está en MERVAL_SET       → argentino, agregar .BA
      - Caso contrario           → global (USD), usar tal cual
    """
    up = sym.upper().strip()
    if up.endswith('.BA'):
        return up, True
    if up in MERVAL_SET:
        return up + '.BA', True
    return up, False


# ─── INDICADORES TÉCNICOS ─────────────────────────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    delta  = np.diff(closes)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
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
    try:
        if len(hist) < period * 2 + 1:
            return None
        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)
        close = hist['Close'].values.astype(float)
        n     = len(close)
        tr    = np.zeros(n)
        pdm   = np.zeros(n)
        ndm   = np.zeros(n)
        for i in range(1, n):
            hl  = high[i] - low[i]
            hpc = abs(high[i] - close[i-1])
            lpc = abs(low[i]  - close[i-1])
            tr[i]  = max(hl, hpc, lpc)
            up     = high[i] - high[i-1]
            down   = low[i-1] - low[i]
            pdm[i] = up   if (up > down and up > 0)   else 0.0
            ndm[i] = down if (down > up and down > 0) else 0.0
        atr  = np.zeros(n); apdm = np.zeros(n); andm = np.zeros(n)
        atr[period]  = np.sum(tr[1:period+1])
        apdm[period] = np.sum(pdm[1:period+1])
        andm[period] = np.sum(ndm[1:period+1])
        for i in range(period+1, n):
            atr[i]  = atr[i-1]  - atr[i-1]/period  + tr[i]
            apdm[i] = apdm[i-1] - apdm[i-1]/period + pdm[i]
            andm[i] = andm[i-1] - andm[i-1]/period + ndm[i]
        pdi     = np.where(atr > 0, 100 * apdm / atr, 0.0)
        ndi     = np.where(atr > 0, 100 * andm / atr, 0.0)
        dx      = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)
        adx_arr = np.zeros(n)
        adx_arr[2*period] = np.mean(dx[period:2*period+1])
        for i in range(2*period+1, n):
            adx_arr[i] = (adx_arr[i-1] * (period-1) + dx[i]) / period
        result = adx_arr[-1]
        return round(float(result), 2) if result > 0 else None
    except Exception:
        return None

def calc_squeeze_momentum(hist, length=20, mult=2.0, length_kc=20, mult_kc=1.5):
    """TTM Squeeze Momentum — LazyBear method."""
    try:
        if len(hist) < length * 2:
            return None
        close = hist['Close'].values.astype(float)
        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)

        s      = pd.Series(close)
        basis  = s.rolling(length).mean()
        dev    = mult * s.rolling(length).std(ddof=0)
        upper_bb = basis + dev
        lower_bb = basis - dev

        hl      = pd.Series(high - low)
        hc      = pd.Series(np.abs(high - np.roll(close, 1)))
        lc      = pd.Series(np.abs(low  - np.roll(close, 1)))
        tr_s    = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        tr_s.iloc[0] = 0
        ma      = s.rolling(length_kc).mean()
        atr_kc  = tr_s.rolling(length_kc).mean()
        upper_kc = ma + mult_kc * atr_kc
        lower_kc = ma - mult_kc * atr_kc

        sqz_on  = bool(lower_bb.iloc[-1] > lower_kc.iloc[-1] and upper_bb.iloc[-1] < upper_kc.iloc[-1])
        sqz_off = bool(lower_bb.iloc[-2] > lower_kc.iloc[-2] and upper_bb.iloc[-2] < upper_kc.iloc[-2]) and not sqz_on

        highest_high = pd.Series(high).rolling(length).max()
        lowest_low   = pd.Series(low).rolling(length).min()
        delta = s - (highest_high + lowest_low) / 2 + basis

        def linreg_last(series, ln):
            arr = series.dropna().values
            if len(arr) < ln:
                return None
            y = arr[-ln:]
            x = np.arange(ln)
            coeffs = np.polyfit(x, y, 1)
            return float(np.polyval(coeffs, ln - 1))

        mom      = linreg_last(delta, length)
        mom_prev = linreg_last(delta.shift(1), length)
        if mom is None:
            return None
        return {
            'sqzOn':     sqz_on,
            'sqzOff':    sqz_off,
            'sqzMom':    round(mom, 4),
            'sqzMomPrev': round(mom_prev, 4) if mom_prev else None,
        }
    except Exception:
        return None

def calc_pe_avg(info: dict) -> float | None:
    """
    P/E promedio del año: mejor aproximación posible con datos de Yahoo Finance.
    Estrategia:
      1. Si hay trailingPE y forwardPE → promedio de ambos
      2. Si solo hay trailingPE       → lo usamos como referencia
      3. Si hay pegRatio y crecimiento → estimamos PE implícito
    El campo 'peAvg' refleja el P/E normalizado/promedio, no solo el spot.
    """
    try:
        trailing = info.get('trailingPE')
        forward  = info.get('forwardPE')
        peg      = info.get('pegRatio')
        eps_growth = info.get('earningsGrowth') or info.get('revenueGrowth')

        # Caso ideal: tenemos ambos
        if trailing and forward and trailing > 0 and forward > 0:
            # Peso mayor al trailing (dato real vs estimado)
            pe_avg = round((trailing * 0.6 + forward * 0.4), 2)
            return pe_avg

        # Solo trailing → usarlo
        if trailing and trailing > 0:
            return round(float(trailing), 2)

        # Solo forward
        if forward and forward > 0:
            return round(float(forward), 2)

        return None
    except Exception:
        return None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.route('/quote', methods=['GET'])
def get_quote():
    symbol_input = request.args.get('symbol', '').upper().strip()
    if not symbol_input:
        return jsonify({'error': 'No symbol provided'}), 400

    yf_symbol, is_ar = resolve_symbol(symbol_input)

    try:
        ticker = yf.Ticker(yf_symbol)

        # Historial 1 año — necesario para EMA200, momentum 12m, squeeze
        hist = ticker.history(period='1y', interval='1d', auto_adjust=True)

        if hist.empty or len(hist) < 5:
            return jsonify({'error': f'No data for {yf_symbol}'}), 404

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
        mom6        = momentum(closes, 126)   # ~6 meses
        mom12       = momentum(closes, 252)   # ~12 meses (anual)

        # Squeeze Momentum (LazyBear TTM)
        sqz = calc_squeeze_momentum(hist)

        # Volúmenes
        vol_today = int(volumes[-1]) if volumes else 0
        avg_vol20 = int(np.mean(volumes[-20:])) if len(volumes) >= 20 else vol_today

        # 52w high / sparkline
        high52         = round(float(max(closes)), 4)
        dist_from_high = pct_dist(price, high52)
        spark          = [round(c, 2) for c in closes[-20:]]

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
        pe_avg     = calc_pe_avg(info)
        beta       = safe('beta')
        div_yield  = safe('dividendYield')
        if div_yield is not None:
            div_yield = round(div_yield * 100, 4)
        currency   = safe('currency', 'ARS' if is_ar else 'USD')
        short_name = safe('shortName', symbol_input)

        result = {
            # Identificación — SIEMPRE devolvemos el símbolo SIN .BA
            'symbol':       symbol_input,
            'shortName':    short_name,
            'currency':     currency,
            # Precio
            'price':        price,
            'prevClose':    prev_close,
            'chgPct':       chg_pct,
            # Técnicos
            'rsi':          rsi,
            'adx':          adx,
            'ema50':        ema50,
            'ema200':       ema200,
            'sma50':        sma50,
            'sma200':       sma200,
            'distEma50':    dist_ema50,
            'distEma200':   dist_ema200,
            'mom6m':        mom6,
            'mom12m':       mom12,           # ← NUEVO: momentum anual
            # Squeeze
            'sqzOn':        sqz['sqzOn']      if sqz else None,
            'sqzOff':       sqz['sqzOff']     if sqz else None,
            'sqzMom':       sqz['sqzMom']     if sqz else None,
            'sqzMomPrev':   sqz['sqzMomPrev'] if sqz else None,
            # Volumen
            'volume':       vol_today,
            'avgVol20':     avg_vol20,
            # 52w / spark
            'high52':       high52,
            'distFromHigh': dist_from_high,
            'spark':        spark,
            # Fundamentals
            'marketCap':    market_cap,
            'pe':           round(pe_ratio, 2) if pe_ratio else None,
            'peAvg':        pe_avg,          # ← NUEVO: P/E promedio año
            'beta':         round(beta, 3) if beta else None,
            'divYield':     div_yield,
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/precio', methods=['GET'])
def get_precio():
    """Fetch rápido: precio, cambio%, volumen. Usado por auto-refresh."""
    symbol_input = request.args.get('symbol', '').upper().strip()
    if not symbol_input:
        return jsonify({'error': 'No symbol'}), 400

    yf_symbol, _ = resolve_symbol(symbol_input)

    try:
        ticker = yf.Ticker(yf_symbol)
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
            'symbol':    symbol_input,
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
    print("\n" + "="*55)
    print("  NEXUS SCANNER — Proxy Server")
    print("="*55)
    print("  URL: http://localhost:5000")
    print()
    print("  ACCIONES ARGENTINAS SIN ADR:")
    print("  MOLI, MORI, BYMA, etc. → busca MOLI.BA en Yahoo Finance")
    print("  El frontend los recibe sin sufijo .BA")
    print()
    print("  CAMPOS NUEVOS EN /quote:")
    print("    mom12m → momentum anual (~252 días hábiles)")
    print("    peAvg  → P/E promedio trailing/forward ponderado")
    print("    sqz*   → TTM Squeeze Momentum (LazyBear)")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
