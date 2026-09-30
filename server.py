"""
NEXUS SCANNER — Servidor Proxy Local
=====================================
Instalación (una sola vez):
    pip install yfinance flask flask-cors pandas numpy requests firebase-admin

Uso:
    python server.py

Luego abrí market-scanner.html en tu browser.
El servidor corre en http://localhost:5000
"""

import os
import json
import re
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)  # Permite que el HTML local llame al servidor

# ─── MONETIZACIÓN: MERCADO PAGO + FIREBASE ADMIN ──────────────────────────────
# Variables de entorno necesarias (configurar en Railway → Variables):
#   MP_ACCESS_TOKEN       -> Access Token de producción de tu cuenta Mercado Pago
#   PRECIO_PREMIUM_ARS    -> 9000 (opcional, default abajo)
#   APP_URL               -> https://dubzmarkets.com (para las urls de retorno del checkout)
#   FIREBASE_SERVICE_ACCOUNT_JSON -> contenido completo del JSON de la service account
#                                    de Firebase (Project Settings → Service Accounts →
#                                    Generate new private key), pegado como texto plano.
MP_ACCESS_TOKEN    = os.environ.get('MP_ACCESS_TOKEN', '')
PRECIO_PREMIUM_ARS = float(os.environ.get('PRECIO_PREMIUM_ARS', '9000'))
APP_URL            = os.environ.get('APP_URL', 'https://dubzmarkets.com')

# ─── ADMIN: otorgar trials manualmente (herramienta trial.html) ──────────────
#   ADMIN_SECRET -> clave que se pide en el formulario admin (configurar en Railway → Variables)
ADMIN_SECRET = os.environ.get('ADMIN_SECRET', '')

# ─── RENDI AI: asistente financiero ────────────────────────────────────────
#   ANTHROPIC_API_KEY -> tu API key de Anthropic (configurar en Railway → Variables)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
AI_MODEL          = os.environ.get('AI_MODEL', 'claude-haiku-4-5-20251001')  # se puede cambiar desde Railway sin tocar el código
AI_MAX_TOKENS     = int(os.environ.get('AI_MAX_TOKENS', '2500'))
AI_WEB_SEARCH     = os.environ.get('AI_WEB_SEARCH', '1') != '0'   # poner AI_WEB_SEARCH=0 para apagar la búsqueda web
AI_WEB_MAX_USES   = int(os.environ.get('AI_WEB_MAX_USES', '3'))   # búsquedas web máximas por pregunta (cada una tiene costo)
AI_FREE_LIMIT     = 2  # preguntas gratis antes de pedir upgrade a premium

# Palabras cortas en mayúsculas que NO son tickers, para no confundirlas al detectarlos en la pregunta
_AI_STOPWORDS_TICKERLIKE = {
    'A','I','Y','O','U','EL','LA','LO','EN','DE','UN','ES','SI','NO','MI','TU','SE','TE','ME',
    'SU','AL','OK','IA','VS','ETC','PBI','USD','ARS','CEO','CFO','ROE','ROA','EPS','PER','ETF',
    'PDF','CSV','P&L','IVA','AFIP','BCRA','MEP','CCL',
}

def find_candidate_tickers(text):
    """Busca posibles tickers (palabras en mayúsculas, 2 a 6 letras, con o sin sufijo .BA)
    dentro de la pregunta del usuario. Es heurístico: los que no sean tickers válidos
    simplemente no van a devolver datos en fetch_fundamentals() y se descartan solos."""
    words = re.findall(r'\b[A-ZÑ]{1,6}(?:\.BA)?\b', (text or '').upper())
    out = []
    for w in words:
        base = w.replace('.BA', '')
        if base in _AI_STOPWORDS_TICKERLIKE or len(base) < 2:
            continue
        if w not in out:
            out.append(w)
    return out[:3]  # como mucho 3, para no demorar la respuesta


def fetch_fundamentals(ticker):
    """Trae datos de mercado y, si están disponibles, fundamentales de yfinance
    para un ticker. Usa history() como base (el mismo endpoint que ya usan /quote
    y /precio, confiable desde servidores cloud) y trata a .info como un "bonus"
    de ratios/sector/resumen que puede fallar sin romper el resto — Yahoo suele
    bloquear .info desde IPs de datacenter, pero no history()."""
    try:
        t = yf.Ticker(ticker)

        precio, max52, min52 = None, None, None
        rend, vol_hoy, vol_prom20 = None, None, None
        try:
            hist = t.history(period='1y', interval='1d', auto_adjust=True)
            if not hist.empty:
                closes = hist['Close'].dropna()
                precio = round(float(closes.iloc[-1]), 4)
                max52  = round(float(closes.max()), 4)
                min52  = round(float(closes.min()), 4)

                def _ret(n):
                    return round((float(closes.iloc[-1]) / float(closes.iloc[-1 - n]) - 1) * 100, 2) if len(closes) > n else None
                rend = {'1d': _ret(1), '5d': _ret(5), '1m': _ret(21), '3m': _ret(63), '6m': _ret(126),
                        '1y': _ret(len(closes) - 1) if len(closes) > 200 else None}
                try:
                    vol_hoy    = _num(hist['Volume'].iloc[-1])
                    vol_prom20 = _num(hist['Volume'].tail(20).mean())
                except Exception:
                    pass
        except Exception as e:
            print('[Rendi AI] history() falló para', ticker, ':', e)

        info = {}
        try:
            info = t.info or {}
        except Exception as e:
            print('[Rendi AI] .info falló para', ticker, '(seguimos solo con precio/rango):', e)

        nombre = info.get('longName') or info.get('shortName')
        if precio is None and not nombre:
            return None  # no hay nada real para este "ticker" -> probablemente no es uno

        return {
            'ticker':                    ticker,
            'nombre':                    nombre,
            'sector':                    info.get('sector'),
            'industria':                 info.get('industry'),
            'precioActual':              precio or info.get('currentPrice') or info.get('regularMarketPrice'),
            'rendimientosPct':           rend,
            'volumenHoy':                vol_hoy,
            'volumenPromedio20d':        vol_prom20,
            'moneda':                    info.get('currency'),
            'marketCap':                 info.get('marketCap'),
            'per_trailing':              info.get('trailingPE'),
            'per_forward':               info.get('forwardPE'),
            'priceToBook':               info.get('priceToBook'),
            'dividendYield':             info.get('dividendYield'),
            'margenNeto':                info.get('profitMargins'),
            'margenOperativo':           info.get('operatingMargins'),
            'roe':                       info.get('returnOnEquity'),
            'crecimientoIngresosYoY':    info.get('revenueGrowth'),
            'deudaSobrePatrimonio':      info.get('debtToEquity'),
            'maxUltimas52Semanas':       max52 or info.get('fiftyTwoWeekHigh'),
            'minUltimas52Semanas':       min52 or info.get('fiftyTwoWeekLow'),
            'precioObjetivoAnalistas':   info.get('targetMeanPrice'),
            'recomendacionAnalistas':    info.get('recommendationKey'),
            'resumenNegocio':            (info.get('longBusinessSummary') or '')[:500],
        }
    except Exception as e:
        print('[Rendi AI] fetch_fundamentals excepción inesperada para', ticker, ':', e)
        return None



# ═══════════════════════════════════════════════════════════════════════════
#  RENDI AI · FUENTES DE DATOS EXTRA (earnings, noticias, analistas) + TOOLS
# ═══════════════════════════════════════════════════════════════════════════
def _num(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return round(f, 4)
    except Exception:
        return None


def _fetch_earnings(t):
    """Últimos balances trimestrales + fechas de earnings (estimado vs reportado)."""
    out = {}
    try:
        qi = t.quarterly_income_stmt
        if qi is not None and not qi.empty:
            filas = {}
            for label, key in (('Total Revenue', 'ingresos'), ('Gross Profit', 'utilidadBruta'),
                               ('Operating Income', 'resultadoOperativo'), ('Net Income', 'resultadoNeto'),
                               ('Diluted EPS', 'epsDiluido')):
                if label in qi.index:
                    ser = qi.loc[label].dropna().head(5)
                    filas[key] = {str(c)[:10]: _num(v) for c, v in ser.items()}
            if filas:
                out['ultimosTrimestres'] = filas
    except Exception as e:
        print('[Rendi AI] quarterly_income_stmt falló:', e)
    try:
        ed = t.get_earnings_dates(limit=8)
        if ed is not None and not ed.empty:
            lst = []
            for idx, row in ed.iterrows():
                lst.append({
                    'fecha':        str(idx)[:10],
                    'epsEstimado':  _num(row.get('EPS Estimate')),
                    'epsReportado': _num(row.get('Reported EPS')),
                    'sorpresaPct':  _num(row.get('Surprise(%)')),
                })
            out['fechasEarnings'] = lst[:6]   # epsReportado vacío = todavía no reportó
    except Exception as e:
        print('[Rendi AI] get_earnings_dates falló:', e)
    try:
        cal = t.calendar
        if isinstance(cal, dict) and cal:
            fe = cal.get('Earnings Date')
            if fe:
                out['proximoEarnings'] = [str(x) for x in fe] if isinstance(fe, (list, tuple)) else str(fe)
            if cal.get('Earnings Average') is not None:
                out['epsEstimadoProximo'] = _num(cal.get('Earnings Average'))
            if cal.get('Revenue Average') is not None:
                out['ingresosEstimadosProximo'] = _num(cal.get('Revenue Average'))
    except Exception as e:
        print('[Rendi AI] calendar falló:', e)
    return out


def _fetch_news(t, limit=6):
    items = []
    try:
        for n in (t.news or [])[:limit]:
            c = n.get('content') if isinstance(n.get('content'), dict) else None
            if c:   # formato nuevo de yfinance
                items.append({'titulo': c.get('title'),
                              'fuente': (c.get('provider') or {}).get('displayName'),
                              'fecha':  str(c.get('pubDate') or '')[:10],
                              'resumen': (c.get('summary') or '')[:200]})
            else:   # formato viejo
                ts = n.get('providerPublishTime')
                items.append({'titulo': n.get('title'), 'fuente': n.get('publisher'),
                              'fecha': datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') if ts else None})
    except Exception as e:
        print('[Rendi AI] news falló:', e)
    return items


def _fetch_analysts(t):
    out = {}
    try:
        apt = t.analyst_price_targets
        if isinstance(apt, dict) and apt:
            out['precioObjetivo'] = {k: _num(v) for k, v in apt.items()}
    except Exception as e:
        print('[Rendi AI] analyst_price_targets falló:', e)
    try:
        rs = t.recommendations_summary
        if rs is not None and not rs.empty:
            row = rs.iloc[0].to_dict()
            out['recomendaciones'] = {k: (v if isinstance(v, str) else _num(v)) for k, v in row.items()}
    except Exception as e:
        print('[Rendi AI] recommendations_summary falló:', e)
    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            out['cambiosRecientes'] = [
                {'fecha': str(i)[:10], 'firma': r.get('Firm'), 'de': r.get('FromGrade'),
                 'a': r.get('ToGrade'), 'accion': r.get('Action')}
                for i, r in ud.head(5).iterrows()
            ]
    except Exception as e:
        print('[Rendi AI] upgrades_downgrades falló:', e)
    return out


_AI_PACK_CACHE = {}
_AI_PACK_TTL   = 180  # segundos: evita pedirle lo mismo a Yahoo en preguntas seguidas


def fetch_market_pack(ticker):
    """Paquete completo de un ticker: precio/rendimientos/volumen + fundamentals +
    balances y earnings + noticias + analistas. Cada fuente falla de forma independiente."""
    ticker = (ticker or '').strip().upper()
    if not re.match(r'^[A-Z0-9.^=\-]{1,14}$', ticker):
        return None
    c = _AI_PACK_CACHE.get(ticker)
    if c and time.time() - c[0] < _AI_PACK_TTL:
        return c[1]

    base = fetch_fundamentals(ticker)
    if not base:
        return None
    pack = dict(base)
    try:
        t = yf.Ticker(ticker)
        ex = ThreadPoolExecutor(max_workers=3)
        futs = {
            'earnings': ex.submit(_fetch_earnings, t),
            'noticias': ex.submit(_fetch_news, t),
            'analistas': ex.submit(_fetch_analysts, t),
        }
        deadline = time.time() + 18
        for k, f in futs.items():
            try:
                pack[k] = f.result(timeout=max(0.1, deadline - time.time()))
            except Exception as e:
                print('[Rendi AI]', k, 'no llegó a tiempo para', ticker, ':', e)
        ex.shutdown(wait=False)
    except Exception as e:
        print('[Rendi AI] fetch_market_pack extras fallaron para', ticker, ':', e)

    pack = {k: v for k, v in pack.items() if v not in (None, {}, [], '')}
    _AI_PACK_CACHE[ticker] = (time.time(), pack)
    return pack


AI_TOOLS = [{
    'name': 'datos_ticker',
    'description': (
        'Trae datos en vivo de un activo: precio, rendimientos (1d a 1 año), volumen, fundamentals '
        '(PER, márgenes, deuda, ROE, crecimiento), últimos balances trimestrales (ingresos, resultado, EPS), '
        'fechas de earnings con EPS estimado vs reportado y sorpresa, próximo earnings, noticias recientes y '
        'opinión de analistas (precio objetivo, recomendaciones). Usala para CUALQUIER empresa o activo que '
        'el usuario mencione y que no esté ya en los datos en vivo pre-cargados. Acepta tickers de EEUU '
        '(MU, AAPL, NVDA) y argentinos con sufijo .BA (GGAL.BA, YPFD.BA). Si el usuario nombra la empresa '
        'sin ticker, deducí el ticker vos mismo y llamala; podés llamarla varias veces para comparar activos.'),
    'input_schema': {
        'type': 'object',
        'properties': {'ticker': {'type': 'string', 'description': 'Ticker de Yahoo Finance, ej. MU o GGAL.BA'}},
        'required': ['ticker'],
    },
}]

AI_WEB_TOOL = {'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': AI_WEB_MAX_USES}


def run_ai_tool(name, inp):
    if name == 'datos_ticker':
        tk = str((inp or {}).get('ticker') or '').strip().upper()
        pack = fetch_market_pack(tk)
        if not pack:
            return 'No se encontraron datos para "%s". Probá con otro ticker (EEUU: MU, AAPL; Argentina: GGAL.BA).' % tk
        return json.dumps(pack, ensure_ascii=False, default=str)[:7000]
    return 'Herramienta desconocida: ' + str(name)


class AIError(Exception):
    def __init__(self, status, detail):
        super().__init__('Anthropic devolvió %s' % status)
        self.status = status
        self.detail = detail


def build_ai_messages(historial, pregunta):
    """Arma la conversación para el modelo: historial previo (alternando user/assistant,
    empezando por user) + la pregunta actual."""
    msgs = []
    for m in (historial or [])[-10:]:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        text = str(m.get('content') or '').strip()[:1500]
        if role not in ('user', 'assistant') or not text:
            continue
        if msgs and msgs[-1]['role'] == role:
            msgs[-1]['content'] += '\n' + text
        else:
            msgs.append({'role': role, 'content': text})
    while msgs and msgs[0]['role'] != 'user':
        msgs.pop(0)
    if msgs and msgs[-1]['role'] == 'user':
        msgs[-1]['content'] += '\n' + pregunta
    else:
        msgs.append({'role': 'user', 'content': pregunta})
    return msgs


def call_anthropic(system_prompt, messages, use_web=True):
    """Llama a Claude con herramientas: datos_ticker (la ejecutamos nosotros) y web_search
    (la ejecuta Anthropic). Itera hasta que el modelo da la respuesta final."""
    tools = list(AI_TOOLS) + ([AI_WEB_TOOL] if use_web else [])
    msgs = list(messages)
    content = []
    for _ in range(8):
        r = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key':         ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'Content-Type':      'application/json',
            },
            json={'model': AI_MODEL, 'max_tokens': AI_MAX_TOKENS, 'system': system_prompt,
                  'messages': msgs, 'tools': tools},
            timeout=75,
        )
        if r.status_code == 400 and AI_WEB_TOOL in tools:
            # búsqueda web no habilitada en la cuenta/modelo: seguimos sin ella en vez de fallar
            print('[Rendi AI] 400 con web_search, reintento sin búsqueda web:', r.text[:300])
            tools = list(AI_TOOLS)
            continue
        if not r.ok:
            raise AIError(r.status_code, r.text)
        res = r.json()
        content = res.get('content', [])
        stop = res.get('stop_reason')
        if stop == 'tool_use':
            msgs.append({'role': 'assistant', 'content': content})
            results = []
            for b in content:
                if b.get('type') == 'tool_use':
                    try:
                        out = run_ai_tool(b.get('name'), b.get('input') or {})
                    except Exception as e:
                        out = 'Error al ejecutar la herramienta: ' + str(e)
                    results.append({'type': 'tool_result', 'tool_use_id': b.get('id'), 'content': out})
            msgs.append({'role': 'user', 'content': results})
            continue
        if stop == 'pause_turn':   # turno largo con búsquedas web: se continúa
            msgs.append({'role': 'assistant', 'content': content})
            continue
        return content
    return content


def extract_final_text(content):
    """Texto final del modelo, sin los comentarios que hace antes de usar las herramientas."""
    last_tool = -1
    for i, b in enumerate(content):
        if b.get('type') in ('server_tool_use', 'web_search_tool_result', 'tool_use'):
            last_tool = i
    txt = ''.join(b.get('text', '') for b in content[last_tool + 1:] if b.get('type') == 'text').strip()
    if not txt:
        txt = ''.join(b.get('text', '') for b in content if b.get('type') == 'text').strip()
    return txt


def extract_json_loose(raw_text):
    """Intenta parsear JSON estricto; si el modelo agregó texto extra antes/después
    (pasa a veces), busca el primer bloque {...} y lo prueba también."""
    cleaned = (raw_text or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        if cleaned[:4].lower() == 'json':
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0), strict=False)
        except Exception:
            pass
    return None

_db = None
def get_firestore():
    """Inicializa Firebase Admin una sola vez y devuelve el cliente Firestore."""
    global _db
    if _db is not None:
        return _db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        cred_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
        if not cred_json:
            print('[MP] FIREBASE_SERVICE_ACCOUNT_JSON no configurado — no se puede activar premium automáticamente.')
            return None
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
        return _db
    except Exception as e:
        print('[MP] Error inicializando Firebase Admin:', e)
        return None


def set_user_plan(uid, plan, extra=None):
    """Escribe el plan (free/premium) del usuario en Firestore."""
    db = get_firestore()
    if db is None:
        return False
    data = {'plan': plan}
    if extra:
        data.update(extra)
    db.collection('users').document(uid).set(data, merge=True)
    return True


def get_user_plan_info(uid):
    """Lee plan/premiumUntil/aiQuestionsUsed del usuario desde Firestore."""
    db = get_firestore()
    if db is None:
        return {'plan': 'free', 'premiumUntil': None, 'aiQuestionsUsed': 0}
    doc = db.collection('users').document(uid).get()
    if not doc.exists:
        return {'plan': 'free', 'premiumUntil': None, 'aiQuestionsUsed': 0}
    d = doc.to_dict() or {}
    return {
        'plan': d.get('plan', 'free'),
        'premiumUntil': d.get('premiumUntil'),
        'aiQuestionsUsed': int(d.get('aiQuestionsUsed', 0) or 0),
    }

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

def calc_weekly_volume(hist):
    """A partir del historial diario ya descargado (sin pedir nada nuevo a
    Yahoo), agrupa por semana calendario y devuelve:
      - vol_week: volumen acumulado de la semana en curso (puede estar a mitad)
      - avg_vol_week: promedio de volumen semanal de las semanas COMPLETAS
        anteriores (excluye la semana en curso para no ensuciar el promedio
        con una semana a la mitad), sobre las últimas ~20 semanas.
    """
    try:
        weekly = hist['Volume'].resample('W').sum()
        weekly = weekly[weekly > 0]
        if weekly.empty:
            return None, None
        vol_week = int(weekly.iloc[-1])
        prev_weeks = weekly.iloc[-21:-1] if len(weekly) > 1 else weekly.iloc[0:0]
        avg_vol_week = int(prev_weeks.mean()) if len(prev_weeks) else vol_week
        return vol_week, avg_vol_week
    except Exception:
        return None, None


def calc_volatility_annualized(closes, period=20):
    """Volatilidad real: desvío estándar de los retornos diarios (log),
    anualizado y expresado en %. A diferencia del beta, no depende de un
    dato externo de terceros — se calcula con los mismos cierres que ya
    usamos para RSI/EMA, así que no agrega ningún request extra."""
    try:
        window = closes[-(period + 1):] if len(closes) >= period + 1 else closes
        if len(window) < 5:
            return None
        log_rets = np.diff(np.log(window))
        vol = float(np.std(log_rets, ddof=1) * np.sqrt(252) * 100)
        return round(vol, 2)
    except Exception:
        return None


# ─── BETA CALCULADO LOCALMENTE (cross-check del beta de Yahoo) ────────────────
# El campo info['beta'] que trae yfinance es un dato de Yahoo a 5 años con
# frecuencia MENSUAL, calculado contra el S&P 500 — y para varios ADRs
# argentinos o tickers .BA viene vacío (None) directamente porque Yahoo no
# tiene suficiente historial mensual "limpio" para esos papeles. Para no
# depender solo de eso, calculamos un beta propio con los mismos precios
# DIARIOS que ya bajamos para el resto de los indicadores (1 año, sin pedir
# nada extra por ticker), contra un índice de referencia. El índice se
# cachea en memoria (10 min) para no volver a pedirlo en cada request.
_BENCH_CACHE = {}
_BENCH_TTL_SECONDS = 600  # 10 minutos

def _get_benchmark_closes(bench_symbol, period='1y'):
    now = datetime.now().timestamp()
    cached = _BENCH_CACHE.get(bench_symbol)
    if cached and (now - cached[0]) < _BENCH_TTL_SECONDS:
        return cached[1]
    try:
        bh = yf.Ticker(bench_symbol).history(period=period, interval='1d', auto_adjust=True)
        bclose = bh['Close'].dropna()
        if bclose.empty:
            return cached[1] if cached else None
        _BENCH_CACHE[bench_symbol] = (now, bclose)
        return bclose
    except Exception:
        return cached[1] if cached else None

def calc_beta_local(close_series, symbol):
    """Beta propio = cov(retornos del activo, retornos del índice) / var(retornos
    del índice), sobre ~1 año de cierres diarios. Índice de referencia: ^MERV
    para tickers .BA (mercado argentino), SPY para el resto (ADRs y US)."""
    try:
        bench_symbol = '^MERV' if symbol.upper().endswith('.BA') else 'SPY'
        bench_closes = _get_benchmark_closes(bench_symbol)
        if bench_closes is None or len(bench_closes) < 30:
            return None
        df = pd.DataFrame({'a': close_series, 'b': bench_closes}).dropna()
        if len(df) < 30:
            return None
        ra = df['a'].pct_change().dropna()
        rb = df['b'].pct_change().dropna()
        aligned = pd.DataFrame({'ra': ra, 'rb': rb}).dropna()
        if len(aligned) < 30:
            return None
        cov = np.cov(aligned['ra'], aligned['rb'])[0][1]
        var = np.var(aligned['rb'])
        if not var:
            return None
        return round(float(cov / var), 3)
    except Exception:
        return None


# ─── ENDPOINT PRINCIPAL ───────────────────────────────────────────────────────

def calc_squeeze_momentum(hist, length=20, mult_bb=2.0, mult_kc=1.5):
    """
    TTM Squeeze Momentum — LazyBear method.
    Devuelve: { sqzOn, sqzOff, sqzMom, sqzMomPrev }
      sqzOn   = True  → squeeze activo (Bollinger dentro de Keltner = compresión)
      sqzOff  = True  → squeeze se acaba de liberar (potencial ruptura)
      sqzMom  = valor del momentum (positivo = alcista, negativo = bajista)
      sqzMomPrev = momentum de la vela anterior (para detectar dirección)
    """
    try:
        if len(hist) < length + 5:
            return None

        close = hist['Close'].values.astype(float)
        high  = hist['High'].values.astype(float)
        low   = hist['Low'].values.astype(float)
        n     = len(close)

        # ── Bollinger Bands ──────────────────────────────
        def sma(arr, p):
            return np.array([np.mean(arr[max(0,i-p+1):i+1]) if i >= p-1 else np.nan for i in range(len(arr))])

        def stdev(arr, p):
            return np.array([np.std(arr[max(0,i-p+1):i+1], ddof=0) if i >= p-1 else np.nan for i in range(len(arr))])

        basis = sma(close, length)
        dev   = stdev(close, length)
        bb_upper = basis + mult_bb * dev
        bb_lower = basis - mult_bb * dev

        # ── Keltner Channels ─────────────────────────────
        # True Range para ATR
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        tr[0] = high[0] - low[0]
        atr_kc = sma(tr, length)
        kc_upper = basis + mult_kc * atr_kc
        kc_lower = basis - mult_kc * atr_kc

        # ── Squeeze detection ────────────────────────────
        sqz_on  = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

        # ── Momentum (delta de precio vs media) ──────────
        def highest(arr, p):
            return np.array([np.max(arr[max(0,i-p+1):i+1]) if i >= p-1 else np.nan for i in range(len(arr))])
        def lowest(arr, p):
            return np.array([np.min(arr[max(0,i-p+1):i+1]) if i >= p-1 else np.nan for i in range(len(arr))])

        mid   = (highest(high, length) + lowest(low, length)) / 2
        delta = close - (mid + basis) / 2

        # Regresión lineal del delta (length períodos)
        mom = np.full(n, np.nan)
        for i in range(length - 1, n):
            y  = delta[i-length+1:i+1]
            x  = np.arange(length, dtype=float)
            if np.any(np.isnan(y)):
                continue
            xm = x.mean(); ym = y.mean()
            denom = np.sum((x - xm) ** 2)
            if denom == 0:
                mom[i] = 0.0
                continue
            slope  = np.sum((x - xm) * (y - ym)) / denom
            intercept = ym - slope * xm
            mom[i] = slope * (length - 1) + intercept

        if np.isnan(mom[-1]):
            return None

        return {
            'sqzOn':      bool(sqz_on[-1]),
            'sqzOff':     bool(sqz_off[-1]),
            'sqzMom':     round(float(mom[-1]), 4),
            'sqzMomPrev': round(float(mom[-2]), 4) if not np.isnan(mom[-2]) else None,
        }
    except Exception:
        return None

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

        close_series = hist['Close'].dropna()
        closes = close_series.values.tolist()
        volumes = hist['Volume'].dropna().values.tolist()

        price = round(closes[-1], 4)
        prev_close = round(closes[-2], 4) if len(closes) >= 2 else price
        chg_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0

        # Indicadores técnicos
        rsi    = calc_rsi(closes)
        adx    = calc_adx(hist)
        ema50  = calc_ema(closes, 50)
        ema200 = calc_ema(closes, 200)
        sma50  = calc_sma(closes, 50)
        sma200 = calc_sma(closes, 200)
        dist_ema50  = pct_dist(price, ema50)
        dist_ema200 = pct_dist(price, ema200)
        mom3 = momentum(closes, 63)   # ~3 meses
        mom6 = momentum(closes, 126)  # ~6 meses

        # Squeeze Momentum (TTM LazyBear)
        sqz = calc_squeeze_momentum(hist) or {}

        # Volúmenes
        vol_today = int(volumes[-1]) if volumes else 0
        avg_vol20 = int(np.mean(volumes[-20:])) if len(volumes) >= 20 else vol_today
        vol_week, avg_vol_week = calc_weekly_volume(hist)
        vol_annualized = calc_volatility_annualized(closes, period=20)

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
        beta_calc   = calc_beta_local(close_series, symbol)
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

        # ── S de CANSLIM: Supply & Demand ──────────────────────────────────
        # Float / acciones en circulación (poca oferta = movimientos de precio más marcados)
        float_shares = safe('floatShares')
        shares_outstanding = safe('sharesOutstanding')

        # Volumen en días de suba vs. días de baja (accumulation/distribution),
        # sobre las últimas 20 ruedas — más fiel a "demanda" que el volumen promedio solo.
        up_down_vol_ratio = None
        try:
            n = min(20, len(closes) - 1)
            if n > 5:
                up_vol, down_vol = 0.0, 0.0
                for i in range(len(closes) - n, len(closes)):
                    if closes[i] > closes[i - 1]:
                        up_vol += volumes[i]
                    elif closes[i] < closes[i - 1]:
                        down_vol += volumes[i]
                if down_vol > 0:
                    up_down_vol_ratio = round(up_vol / down_vol, 2)
        except Exception:
            pass

        # ── I de CANSLIM: Institutional Sponsorship ────────────────────────
        held_pct_institutions = None
        try:
            hpi = safe('heldPercentInstitutions')
            if hpi is not None:
                held_pct_institutions = round(float(hpi) * 100, 1)
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
            'volWeek':       vol_week,
            'avgVolWeek':    avg_vol_week,
            'volAnnualized': vol_annualized,
            'high52':        high52,
            'distFromHigh':  dist_from_high,
            'marketCap':     market_cap,
            'pe':            round(pe_ratio, 2) if pe_ratio else None,
            'beta':          round(beta, 3) if beta else None,
            'betaCalc':      beta_calc,
            'divYield':      div_yield,
            'spark':         spark,
            # CANSLIM fundamentals
            'epsGrowthQ':    eps_growth_q,
            'epsGrowthA':    eps_growth_a,
            'epsAccel':      eps_accel,
            'revGrowthQ':    rev_growth_q,
            'roe':           roe,
            'profitMargin':  profit_margin,
            'floatShares':      float_shares,
            'sharesOutstanding': shares_outstanding,
            'upDownVolRatio':   up_down_vol_ratio,
            'heldPctInstitutions': held_pct_institutions,
            # Squeeze Momentum (TTM LazyBear)
            'sqzOn':         sqz.get('sqzOn'),
            'sqzOff':        sqz.get('sqzOff'),
            'sqzMom':        sqz.get('sqzMom'),
            'sqzMomPrev':    sqz.get('sqzMomPrev'),
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/squeeze1h', methods=['GET'])
def get_squeeze_1h():
    """Endpoint liviano: SOLO el TTM Squeeze Momentum calculado en velas de 1 hora.
    A diferencia de /quote, acá NO se descarga 1 año de historial diario ni
    ticker.info/financials (que es lo que hace lento a /quote) — solo se
    piden ~10 días de velas horarias, lo mínimo para el cálculo (length=20).
    Pensado para llamarse en paralelo y ANTES que /quote, así la columna
    de squeeze aparece primero y el resto de los datos se completa después.
    """
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    try:
        ticker = yf.Ticker(symbol)
        # 10d de velas de 1h ≈ 60-70 velas en mercados US (6.5h/día) — de sobra
        # para el largo=20 que usa calc_squeeze_momentum. Para tickers .BA con
        # menos horas de rueda, igual alcanza.
        hist = ticker.history(period='10d', interval='60m', auto_adjust=True)
        if hist.empty or len(hist) < 25:
            return jsonify({'error': 'No data'}), 404

        sqz = calc_squeeze_momentum(hist) or {}
        return jsonify({
            'symbol':     symbol,
            'sqzOn':      sqz.get('sqzOn'),
            'sqzOff':     sqz.get('sqzOff'),
            'sqzMom':     sqz.get('sqzMom'),
            'sqzMomPrev': sqz.get('sqzMomPrev'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/squeezeD', methods=['GET'])
def get_squeeze_daily():
    """Endpoint liviano: SOLO el TTM Squeeze Momentum calculado en velas
    DIARIAS. Existe por la misma razón que /squeeze1h: /quote es lento
    porque además del historial pide ticker.info/financials/quarterly_financials
    (varios round-trips extra a Yahoo), y hasta ahora el squeeze diario venía
    metido adentro de esa respuesta pesada, así que tardaba lo mismo que todo
    lo demás. Acá solo se piden ~6 meses de velas diarias (de sobra para el
    largo=20 del cálculo) y no se toca info/financials — mucho más rápido.
    Pensado para llamarse en paralelo y ANTES que /quote, igual que /squeeze1h.
    """
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='6mo', interval='1d', auto_adjust=True)
        if hist.empty or len(hist) < 25:
            return jsonify({'error': 'No data'}), 404

        sqz = calc_squeeze_momentum(hist) or {}
        return jsonify({
            'symbol':     symbol,
            'sqzOn':      sqz.get('sqzOn'),
            'sqzOff':     sqz.get('sqzOff'),
            'sqzMom':     sqz.get('sqzMom'),
            'sqzMomPrev': sqz.get('sqzMomPrev'),
        })
    except Exception as e:
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


@app.route('/crear-suscripcion', methods=['POST'])
def crear_suscripcion():
    """Crea una suscripción recurrente mensual en Mercado Pago (Preapproval API)
    y devuelve la URL de checkout (init_point) para redirigir al usuario.
    Body esperado: {"uid": "<firebase_uid>", "email": "<email_usuario>"}
    """
    if not MP_ACCESS_TOKEN:
        return jsonify({'error': 'MP_ACCESS_TOKEN no configurado en el servidor'}), 500

    data  = request.get_json(silent=True) or {}
    uid   = data.get('uid')
    email = data.get('email')
    if not uid or not email:
        return jsonify({'error': 'Falta uid o email'}), 400

    payload = {
        "reason": "DUBZ Monitor Premium — Suscripción mensual",
        "auto_recurring": {
            "frequency": 1,
            "frequency_type": "months",
            "transaction_amount": PRECIO_PREMIUM_ARS,
            "currency_id": "ARS"
        },
        "payer_email": email,
        # external_reference nos permite identificar al usuario cuando llega el webhook
        "external_reference": uid,
        "back_url": APP_URL,
        "status": "pending"
    }

    try:
        r = requests.post(
            'https://api.mercadopago.com/preapproval',
            headers={
                'Authorization': f'Bearer {MP_ACCESS_TOKEN}',
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=15
        )
        r.raise_for_status()
        mp_data = r.json()
        return jsonify({
            'init_point': mp_data.get('init_point'),
            'id':         mp_data.get('id')
        })
    except requests.exceptions.HTTPError:
        return jsonify({'error': 'Mercado Pago rechazó la solicitud', 'detalle': r.text}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/webhook-mp', methods=['POST'])
def webhook_mp():
    """Recibe las notificaciones IPN/webhook de Mercado Pago.
    Configurar esta URL en: Mercado Pago → Tu integración → Webhooks
    -> https://TU-DOMINIO/webhook-mp  (eventos: 'subscription_preapproval' y 'subscription_authorized_payment')
    """
    try:
        topic = request.args.get('topic') or request.args.get('type')
        body  = request.get_json(silent=True) or {}
        resource_id = (
            request.args.get('id')
            or body.get('data', {}).get('id')
            or body.get('id')
        )
        if not resource_id:
            return jsonify({'status': 'ignored'}), 200

        if not MP_ACCESS_TOKEN:
            return jsonify({'status': 'ignored', 'reason': 'no MP token'}), 200

        # 'preapproval' = alta/estado de la suscripción.
        # 'authorized_payment' = cada cobro mensual efectivo.
        if topic in ('preapproval', 'subscription_preapproval'):
            r = requests.get(
                f'https://api.mercadopago.com/preapproval/{resource_id}',
                headers={'Authorization': f'Bearer {MP_ACCESS_TOKEN}'}, timeout=15
            )
            info = r.json()
            uid    = info.get('external_reference')
            status = info.get('status')  # authorized | paused | cancelled
            if uid:
                if status == 'authorized':
                    hasta = (datetime.utcnow() + timedelta(days=35)).isoformat()
                    set_user_plan(uid, 'premium', {
                        'premiumUntil': hasta,
                        'mpPreapprovalId': resource_id
                    })
                elif status in ('paused', 'cancelled'):
                    set_user_plan(uid, 'free')

        elif topic in ('authorized_payment', 'subscription_authorized_payment'):
            r = requests.get(
                f'https://api.mercadopago.com/authorized_payments/{resource_id}',
                headers={'Authorization': f'Bearer {MP_ACCESS_TOKEN}'}, timeout=15
            )
            info = r.json()
            if info.get('status') == 'approved':
                preapproval_id = info.get('preapproval_id')
                # Buscar el uid a partir del preapproval (guardado en Firestore al autorizar)
                db = get_firestore()
                if db is not None and preapproval_id:
                    q = db.collection('users').where('mpPreapprovalId', '==', preapproval_id).limit(1).get()
                    for docu in q:
                        hasta = (datetime.utcnow() + timedelta(days=35)).isoformat()
                        set_user_plan(docu.id, 'premium', {'premiumUntil': hasta})

        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        traceback.print_exc()
        # Devolver 200 igual para que MP no reintente indefinidamente por errores nuestros
        return jsonify({'status': 'error', 'detalle': str(e)}), 200


@app.route('/ai/chat', methods=['POST'])
def ai_chat():
    """Asistente de IA financiero (Rendi AI).
    Responde preguntas sobre la cartera/mercado y, si el usuario dicta una
    orden ("compré 10 GGAL a 4800"), la devuelve como JSON estructurado
    para que el frontend la muestre en el modal de carga (con confirmación
    manual del usuario antes de guardarla).

    Body esperado: {"uid": "...", "pregunta": "...", "contexto": "...", "isAdmin": bool,
    "historial": [{"role": "user|assistant", "content": "..."}], "tickersMencionados": [...], "fecha": "..."}
    """
    if not ANTHROPIC_API_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY no configurado en el servidor'}), 500

    data     = request.get_json(silent=True) or {}
    uid      = data.get('uid')
    pregunta = (data.get('pregunta') or '').strip()
    contexto = data.get('contexto') or '(el usuario no tiene posiciones cargadas todavía)'
    is_admin = bool(data.get('isAdmin'))

    if not uid or not pregunta:
        return jsonify({'error': 'Falta uid o pregunta'}), 400

    info          = get_user_plan_info(uid)
    premium_until = info.get('premiumUntil')
    vigente       = True
    if premium_until:
        try:
            vigente = datetime.fromisoformat(str(premium_until).replace('Z', '')) > datetime.utcnow()
        except Exception:
            vigente = True
    es_premium = is_admin or (info['plan'] == 'premium' and vigente)

    if not es_premium and info['aiQuestionsUsed'] >= AI_FREE_LIMIT:
        return jsonify({
            'error':   'limite_free',
            'mensaje': 'Ya usaste tus ' + str(AI_FREE_LIMIT) + ' preguntas gratis a Rendi AI. Con Premium tenés preguntas ilimitadas.'
        }), 403

    # Tickers: los que detectó el frontend (incluye nombres como "micron" -> MU) + los que aparecen en la pregunta
    candidatos = []
    for tk in list(data.get('tickersMencionados') or []) + find_candidate_tickers(pregunta):
        tk = str(tk or '').strip().upper()
        if tk and tk not in candidatos and re.match(r'^[A-Z0-9.^=\-]{1,14}$', tk):
            candidatos.append(tk)
    candidatos = candidatos[:3]

    # Pre-cargar en paralelo el paquete completo (precio, fundamentals, balances, noticias, analistas)
    fundamentales_txt = ''
    if candidatos:
        ex = ThreadPoolExecutor(max_workers=len(candidatos))
        futs = [(tk, ex.submit(fetch_market_pack, tk)) for tk in candidatos]
        for tk, f in futs:
            try:
                datos = f.result(timeout=25)
                if datos:
                    fundamentales_txt += '\n' + tk + ': ' + json.dumps(datos, ensure_ascii=False, default=str)[:6000]
            except Exception as e:
                print('[Rendi AI] pre-carga falló para', tk, ':', e)
        ex.shutdown(wait=False)

    fecha = (data.get('fecha') or datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))

    system_prompt = (
        "Sos Rendi AI, el analista financiero de la app DUBZ Monitor. Ayudás al usuario con su cartera, "
        "con acciones argentinas (Merval) y de EEUU, y con cualquier empresa o activo que tenga ticker. "
        "Respondé siempre en español rioplatense, con datos concretos, completo pero sin relleno.\n\n"
        "Fecha y hora actual: " + str(fecha) + "\n\n"
        "CÓMO CONSEGUIR LOS DATOS (usá todas las fuentes que hagan falta, no te quedes con la primera):\n"
        "1. 'Contexto del usuario' y 'Datos en vivo pre-cargados' (más abajo): vienen del scanner de la app y "
        "de Yahoo Finance. Son actuales y confiables; usalos como fuente de verdad para precios, indicadores "
        "técnicos (RSI, ADX, EMAs, squeeze, volumen), fundamentals, balances recientes, analistas y noticias.\n"
        "2. Herramienta datos_ticker: llamala vos para cualquier activo que el usuario nombre y no esté ya en los "
        "datos pre-cargados. Si nombra la empresa sin ticker (ej. 'Micron'), deducí el ticker (MU) y llamala "
        "directamente. NUNCA le pidas al usuario que confirme un ticker que podés deducir vos.\n"
        "3. Herramienta web_search: para todo lo reciente o que no está en las otras fuentes: resultados o balances "
        "recién publicados, guidance, conference call, noticias del día, datos macro (dólar, inflación, tasas, "
        "riesgo país, Fed, BCRA). Si preguntan por algo de hoy o de las últimas horas, buscá en la web antes de responder.\n"
        "4. Tu conocimiento general, solo para contexto (qué hace la empresa, cómo leer un indicador). Nunca para "
        "cifras actuales.\n\n"
        "REGLAS:\n"
        "- Nunca digas que no tenés datos en vivo sin haber probado antes las herramientas. Si aun así falta algo, "
        "decí exactamente qué falta.\n"
        "- No inventes cifras: cada número sale de los datos, de las herramientas o de la búsqueda. Si dos fuentes "
        "difieren, mencionalo.\n"
        "- Cuando uses la web, nombrá la fuente y la fecha (ej. 'según Reuters, 30/09').\n"
        "- Balances y earnings: ingresos y EPS reportados vs. estimados (sorpresa), márgenes, crecimiento "
        "interanual, guidance si lo hay, reacción del precio y qué está mirando el mercado.\n"
        "- Sobre una acción: combiná lo técnico (tendencia, RSI, ADX, EMAs, volumen, squeeze) con lo fundamental "
        "(valuación, crecimiento, márgenes, deuda, analistas) y el contexto (noticias, sector, macro). Cerrá con "
        "una lectura clara: qué está bien, qué preocupa y qué niveles o eventos mirar.\n"
        "- Si el activo está en la cartera del usuario, relacionalo con su posición (ganancia o pérdida vs. compra).\n"
        "- Es información general, no asesoramiento personalizado ni garantía de rendimiento. Aclaralo en una línea "
        "cuando des una opinión sobre comprar o vender, sin repetirlo en cada mensaje.\n"
        "- Formato: texto plano. La app NO renderiza markdown (no uses **, # ni tablas). Para listas, líneas que "
        "empiecen con '- '. Párrafos cortos separados por una línea en blanco.\n"
        "- Seguí la conversación: si el usuario responde 'sí' o algo corto, continuá con lo que venían hablando; "
        "no saludes de nuevo ni repitas preguntas.\n\n"
        "Si el usuario está DICTANDO UNA ORDEN para cargar en su cartera (ej: 'compré 10 GGAL a 4800', "
        "'vendí 50 AAPL a 220 dólares'), respondé ÚNICA Y EXCLUSIVAMENTE con este JSON, sin texto antes ni "
        "después, sin markdown, sin explicaciones adicionales:\n"
        '{"tipo":"orden","orden":{"accion":"compra|venta","ticker":"...","cantidad":0,"precio":0,'
        '"moneda":"ARS|USD"},"texto":"resumen breve para confirmarle al usuario"}\n\n'
        "Si es una pregunta normal (no una orden), respondé ÚNICA Y EXCLUSIVAMENTE con este JSON, sin texto "
        "antes ni después y sin markdown fuera del campo texto (los saltos de línea dentro de texto van como \\n):\n"
        '{"tipo":"respuesta","texto":"tu respuesta acá"}\n\n'
        "Contexto del usuario (cartera, scanner y mercado, enviado por la app):\n" + contexto + "\n\n" +
        "Datos en vivo pre-cargados:\n" + (fundamentales_txt or "(ninguno; usá la herramienta datos_ticker si hace falta)")
    )

    try:
        content  = call_anthropic(system_prompt, build_ai_messages(data.get('historial'), pregunta), AI_WEB_SEARCH)
        raw_text = extract_final_text(content)

        # El modelo debería devolver JSON puro; si viene con texto extra antes/después
        # (pasa a veces), extract_json_loose intenta salvarlo antes de degradarlo a texto plano.
        parsed = extract_json_loose(raw_text)
        if parsed is None:
            texto = raw_text
            if raw_text.lstrip().startswith('{'):   # JSON cortado o mal cerrado: rescatamos el campo texto
                m = re.search(r'"texto"\s*:\s*"(.*?)(?:"\s*\}\s*)?$', raw_text, re.DOTALL)
                if m:
                    texto = m.group(1).replace('\\n', '\n').replace('\\"', '"')
            parsed = {'tipo': 'respuesta', 'texto': texto}

        # Contabilizar solo a usuarios free (evita writes innecesarios para premium)
        preguntas_restantes = None
        if not es_premium:
            nuevo_count = info['aiQuestionsUsed'] + 1
            db = get_firestore()
            if db is not None:
                db.collection('users').document(uid).set({'aiQuestionsUsed': nuevo_count}, merge=True)
            preguntas_restantes = max(0, AI_FREE_LIMIT - nuevo_count)

        return jsonify({
            'tipo':               parsed.get('tipo', 'respuesta'),
            'texto':              parsed.get('texto', raw_text),
            'orden':              parsed.get('orden'),
            'preguntasRestantes': preguntas_restantes,
        })
    except AIError as e:
        print('[Rendi AI] Anthropic devolvió error', e.status, '->', e.detail)
        return jsonify({'error': 'La API de IA rechazó la solicitud', 'detalle': e.detail}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/admin/grant-trial', methods=['POST'])
def admin_grant_trial():
    """Otorga premium temporal a usuarios ya registrados, a partir de su email.
    Pensado para usarse desde la herramienta interna trial.html.

    Body esperado: {"secret": "...", "emails": ["a@b.com", ...], "days": 14}
    """
    data   = request.get_json(silent=True) or {}
    secret = data.get('secret')

    if not ADMIN_SECRET:
        return jsonify({'error': 'ADMIN_SECRET no configurado en el servidor'}), 500
    if secret != ADMIN_SECRET:
        return jsonify({'error': 'Clave admin incorrecta'}), 401

    emails = data.get('emails')
    if not isinstance(emails, list) or not emails:
        return jsonify({'error': 'Falta la lista de emails'}), 400

    try:
        days = int(data.get('days') or 14)
    except (TypeError, ValueError):
        days = 14

    db = get_firestore()
    if db is None:
        return jsonify({'error': 'Firebase Admin no está configurado en el servidor'}), 500

    try:
        from firebase_admin import auth as fb_auth
    except Exception as e:
        return jsonify({'error': 'firebase_admin no disponible: ' + str(e)}), 500

    premium_until = (datetime.utcnow() + timedelta(days=days)).isoformat()
    resultados = []

    for raw_email in emails:
        email = (raw_email or '').strip()
        if not email:
            continue
        try:
            user = fb_auth.get_user_by_email(email)
            ok = set_user_plan(user.uid, 'premium', {'premiumUntil': premium_until})
            if ok:
                resultados.append({'email': email, 'status': 'ok', 'premiumUntil': premium_until})
            else:
                resultados.append({'email': email, 'status': 'error', 'detalle': 'No se pudo escribir en Firestore'})
        except fb_auth.UserNotFoundError:
            resultados.append({'email': email, 'status': 'error', 'detalle': 'No existe ningún usuario registrado con ese email'})
        except Exception as e:
            resultados.append({'email': email, 'status': 'error', 'detalle': str(e)})

    return jsonify({'resultados': resultados})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'NEXUS SCANNER proxy running'})


@app.route('/')
def serve_index():
    """Sirve el index.html desde la misma carpeta que server.py"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'index.html')


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  NEXUS SCANNER — Proxy Server")
    print("="*50)
    print("  Corriendo en: http://localhost:5000")
    print("  Abrí market-scanner.html en tu browser")
    print("  Ctrl+C para detener")
    print("="*50 + "\n")
    # threaded=True es CLAVE para la velocidad de carga: sin esto, Flask
    # procesa los requests de a UNO por vez, aunque el frontend mande 25
    # en paralelo (Promise.all) — cada /quote hace varios llamados a Yahoo
    # (history + info + financials) y se queda "colgado" esperando red, así
    # que sin threading todos los demás requests esperan en fila detrás de
    # ese. Con threaded=True, Flask abre un hilo por request y sí corren
    # en paralelo de verdad. Si en Railway usás gunicorn en vez de este
    # app.run(), lo equivalente ahí es `gunicorn --workers 2 --threads 8 server:app`.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
