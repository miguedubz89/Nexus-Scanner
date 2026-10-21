"""
NEXUS SCANNER — Servidor Proxy Local
=====================================
Instalación (una sola vez):
    pip install yfinance flask flask-cors pandas numpy requests firebase-admin flask-compress

Uso:
    python server.py

Luego abrí market-scanner.html en tu browser.
El servidor corre en http://localhost:5000
"""

import os
import json
import re
import requests
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
import html as _html
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)  # Permite que el HTML local llame al servidor
try:
    from flask_compress import Compress   # opcional: pip install flask-compress (JSON ~5x más chico)
    Compress(app)
except Exception:
    pass

# ─── CACHE EN MEMORIA (TTL + "single-flight") ────────────────────────────────
# Yahoo es el cuello de botella de todo el scanner. Estos TTL evitan pedir lo
# mismo dos veces: los fundamentals (balances, EPS, ROE...) cambian una vez por
# trimestre, así que se cachean horas; el historial técnico, minutos.
# "Single-flight": si 5 requests piden el mismo ticker a la vez, solo uno va a
# Yahoo y los otros esperan ese resultado (antes cada uno hacía su propio pedido).
TECH_TTL     = int(os.environ.get('TECH_TTL', '120'))        # seg — historial + indicadores
FUND_TTL     = int(os.environ.get('FUND_TTL', str(6*3600)))  # seg — fundamentals (EPS, ROE, institucional)
MARKET_TTL   = int(os.environ.get('MARKET_TTL', '300'))      # seg — SPY/QQQ/VIX/Merval/macro
FG_TTL       = int(os.environ.get('FG_TTL', '600'))          # seg — Fear & Greed
RRG_TTL      = int(os.environ.get('RRG_TTL', '900'))         # seg — rotación sectorial
PRECIO_TTL   = int(os.environ.get('PRECIO_TTL', '15'))       # seg — /precio (auto-refresh)

_CACHE = {}
_CACHE_LOCK = threading.Lock()
_KEY_LOCKS = {}

def cache_get(key, ttl):
    with _CACHE_LOCK:
        v = _CACHE.get(key)
    if v and (time.time() - v[0]) < ttl:
        return v[1]
    return None

def cache_get_stale(key):
    with _CACHE_LOCK:
        v = _CACHE.get(key)
    return v[1] if v else None

def cache_set(key, val):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), val)
        if len(_CACHE) > 5000:   # límite simple de memoria: borra las 800 más viejas
            for k, _ in sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:800]:
                _CACHE.pop(k, None)
    return val

def cached_call(key, ttl, fn):
    """Devuelve el valor cacheado si está fresco; si no, lo calcula UNA sola vez
    (aunque lleguen varios requests juntos). Si Yahoo falla, devuelve el último
    valor viejo que haya en vez de un error."""
    v = cache_get(key, ttl)
    if v is not None:
        return v
    with _CACHE_LOCK:
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        v = cache_get(key, ttl)
        if v is not None:
            return v
        try:
            val = fn()
        except Exception:
            traceback.print_exc()
            val = None
        if val is not None:
            return cache_set(key, val)
        return cache_get_stale(key)

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


def _fetch_news(t, limit=8):
    """Noticias para Rendi AI: las mismas fuentes que el panel de la cartera (yfinance + Yahoo RSS + Google News
    + prensa argentina para .BA). Si todo falla, cae al método viejo (solo yfinance)."""
    sym = getattr(t, 'ticker', None)
    try:
        if sym:
            its = cached_call('news:' + sym, NEWS_TTL, lambda: _news_items(sym)) or []
            if its:
                return [{'titulo': i.get('title'), 'fuente': i.get('source'), 'fecha': str(i.get('ts') or '')[:10],
                         'url': i.get('url')} for i in its[:limit]]
    except Exception as e:
        print('[Rendi AI] news multi-fuente falló:', e)
    return _fetch_news_yf(t, limit)


def _fetch_news_yf(t, limit=6):
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
    # Mismos números que ve el usuario en el scanner (técnicos + CANSLIM), desde el cache
    try:
        _tc = cached_call('tech:' + ticker, TECH_TTL, lambda: _fetch_tech(ticker))
        _fd = cached_call('fund:' + ticker, FUND_TTL, lambda: _fetch_fund(ticker))
        if _tc:
            pack['scanner_tecnico'] = {k: _tc.get(k) for k in (
                'rsi', 'adx', 'distEma50', 'distEma200', 'mom3m', 'mom6m', 'mom12m', 'rsScore',
                'distFromHigh', 'yearHigh', 'yearLow', 'upDownVolRatio', 'sqzOn', 'sqzOff', 'sqzMom', 'volAnnualized', 'betaCalc')}
        if _fd:
            pack['scanner_canslim'] = {k: _fd.get(k) for k in (
                'epsGrowthQ', 'epsGrowthQPrev', 'epsAccel', 'epsGrowthA', 'epsCagr3', 'epsSource', 'revGrowthQ',
                'roe', 'profitMargin', 'heldPctInstitutions', 'floatShares')}
    except Exception as e:
        print('[Rendi AI] datos de scanner no disponibles para', ticker, ':', e)
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
AI_TOOLS.append({
    'name': 'mercado_hoy',
    'description': ('Panorama del mercado AHORA (instantáneo, sin costo de búsqueda): índice Fear & Greed de CNN, '
                    'dirección de S&P 500 / Nasdaq / Russell / Merval con días de distribución, VIX, bono 10A, DXY, '
                    'oro, petróleo, Bitcoin y rotación sectorial (Líder/Mejorando/Debilitándose/Rezagado). '
                    'Usala para preguntas sobre sentimiento, si el mercado está para comprar, qué sectores lideran, etc.'),
    'input_schema': {'type': 'object', 'properties': {}},
})


AI_TOOLS.append({
    'name': 'noticias_mercado',
    'description': ('Titulares recientes del mercado en general (no de una empresa puntual) desde prensa financiera: '
                    'mercado argentino (Merval, bonos, dólar, riesgo país; Ámbito, Cronista, Infobae, iProfesional) o '
                    'Wall Street. Usala para "qué pasó hoy en el mercado", contexto macro o cuando una acción no tiene '
                    'noticias propias.'),
    'input_schema': {'type': 'object', 'properties': {
        'mercado': {'type': 'string', 'enum': ['ar', 'us'], 'description': "'ar' = Argentina (por defecto), 'us' = EEUU"}}},
})


def market_snapshot_text():
    """Resumen compacto (todo sale de caches de 5-15 min) del estado del mercado para la IA:
    Fear & Greed, dirección de índices (la 'M' de CANSLIM), macro y rotación sectorial."""
    lines = []
    try:
        fg = get_fear_greed()
        if fg:
            s = 'Fear & Greed (%s): %s = %s' % (fg.get('source'), fg.get('score'), fg.get('rating'))
            if fg.get('previousClose') is not None:
                s += ' | cierre previo %s, hace 1 semana %s, hace 1 mes %s' % (fg.get('previousClose'), fg.get('week'), fg.get('month'))
            lines.append(s)
    except Exception:
        pass
    try:
        mk = get_market_data()
        if mk:
            lines.append('Índices: ' + '; '.join(
                '%s %+.2f%% hoy, %s (vs EMA200 %s%%%s)' % (
                    i['name'], i['chgPct'], i['status'] or 's/d',
                    ('%+.1f' % i['distEma200']) if i['distEma200'] is not None else '--',
                    (', %d días distribución' % i['distDays']) if i.get('distDays') is not None else '')
                for i in mk['indexes']))
            if mk.get('macro'):
                lines.append('Macro: ' + '; '.join('%s %s (%+.2f%% hoy)' % (m['name'], m['price'], m['chgPct']) for m in mk['macro']))
    except Exception:
        pass
    try:
        rr = get_rrg('daily', 8)
        if rr:
            by = {}
            for s in rr['sectors']:
                by.setdefault(s['quadrant'], []).append('%s (%s)' % (s['name'], s['symbol']))
            lines.append('Rotación sectorial (RRG diario vs SPY, ETFs SPDR): ' + ' | '.join(
                '%s: %s' % (q, ', '.join(by[q])) for q in ('Líder', 'Mejorando', 'Debilitándose', 'Rezagado') if q in by))
    except Exception:
        pass
    try:   # rotación entre países/índices (ETFs en USD vs ACWI)
        ri = get_rrg_idx('daily', 8)
        if ri:
            by = {}
            for s in ri['sectors']:
                by.setdefault(s['quadrant'], []).append('%s (%s, 1M %+.1f%%)' % (s['name'], s['symbol'], s['chg1m'] or 0))
            lines.append('Rotación entre países/índices (RRG diario vs ACWI, ETFs en USD): ' + ' | '.join(
                '%s: %s' % (q, ', '.join(by[q])) for q in ('Líder', 'Mejorando', 'Debilitándose', 'Rezagado') if q in by))
    except Exception:
        pass
    try:   # rotación sectorial del Merval
        ra = get_rrg_ar('daily', 8)
        if ra:
            by = {}
            for s in ra['sectors']:
                by.setdefault(s['quadrant'], []).append(s['name'])
            lines.append('Rotación sectorial Merval (canastas equiponderadas vs MERVAL): ' + ' | '.join(
                '%s: %s' % (q, ', '.join(by[q])) for q in ('Líder', 'Mejorando', 'Debilitándose', 'Rezagado') if q in by))
    except Exception:
        pass
    try:   # dólar y riesgo país
        d = get_dolar()
        if d:
            parts = ['%s $%s' % (k.upper(), d.get(k)) for k in ('oficial', 'mep', 'ccl', 'blue') if d.get(k)]
            if d.get('brechaCclMep') is not None:
                parts.append('brecha CCL/MEP %s%%' % d['brechaCclMep'])
            if d.get('brechaMepOficial') is not None:
                parts.append('brecha MEP/oficial %s%%' % d['brechaMepOficial'])
            if parts:
                lines.append('Dólar: ' + ' | '.join(parts))
        rp = cached_call('riesgo', RIESGO_TTL, _build_riesgo)
        if rp and rp.get('valor') is not None:
            lines.append('Riesgo país: %s pb (%s)%s' % (rp['valor'], rp.get('fecha'),
                         (', var. 1 mes %+.0f pb' % rp['var1m']) if rp.get('var1m') is not None else ''))
    except Exception:
        pass
    return '\n'.join(lines)


def run_ai_tool(name, inp):
    if name == 'noticias_mercado':
        mk = str((inp or {}).get('mercado') or 'ar').lower()
        ar = mk != 'us'
        its = cached_call('news:mkt:%s' % ('ar' if ar else 'us'), NEWS_TTL, lambda: _market_news(ar, 12)) or []
        if not its:
            return 'No se pudieron traer titulares ahora. Probá con web_search.'
        return '\n'.join('- %s (%s, %s)' % (i.get('title'), i.get('source') or 's/f', str(i.get('ts') or '')[:10]) for i in its)
    if name == 'mercado_hoy':
        return market_snapshot_text() or 'No hay datos de mercado disponibles en este momento.'
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
        if r.status_code == 400 and AI_WEB_TOOL in tools and 'web_search' in r.text.lower().replace(' ', '_'):
            # búsqueda web no habilitada en la cuenta/modelo: seguimos sin ella en vez de fallar
            # (solo si el 400 habla de web_search: otros 400 no deben taparse reintentando)
            print('[Rendi AI] 400 con web_search, reintento sin búsqueda web:', r.text[:300])
            tools = list(AI_TOOLS)
            continue
        if not r.ok:
            raise AIError(r.status_code, r.text)
        res = r.json()
        content = res.get('content', [])
        stop = res.get('stop_reason')
        if stop == 'tool_use':
            results = []
            for b in content:
                if b.get('type') == 'tool_use':
                    try:
                        out = run_ai_tool(b.get('name'), b.get('input') or {})
                    except Exception as e:
                        out = 'Error al ejecutar la herramienta: ' + str(e)
                    results.append({'type': 'tool_result', 'tool_use_id': b.get('id'),
                                    'content': out or '(sin datos)'})
            if not results:
                # stop_reason=tool_use pero sin pedidos que ejecutar nosotros: mandar un mensaje de usuario vacío
                # rompe la API ("user messages must have non-empty content"). Se toma como respuesta final.
                print('[Rendi AI] tool_use sin bloques tool_use; bloques recibidos:', [b.get('type') for b in content])
                return content
            msgs.append({'role': 'assistant', 'content': content})
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
    return round((closes[-1] / closes[-days - 1] - 1) * 100, 2)

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

        # +DI / -DI  (errstate: las primeras posiciones tienen atr=0 y generaban warnings en el log)
        with np.errstate(divide='ignore', invalid='ignore'):
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
    def _f():
        bh = yf.Ticker(bench_symbol).history(period=period, interval='1d', auto_adjust=True)
        bclose = bh['Close'].dropna()
        return bclose if not bclose.empty else None
    return cached_call('bench:' + bench_symbol, _BENCH_TTL_SECONDS, _f)

def _naive_dates(s):
    """Serie con índice de fechas sin zona horaria (alinea tickers de distintas bolsas)."""
    idx = pd.to_datetime(s.index)
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep='last')]

def calc_beta_local(close_series, symbol):
    """Beta propio = cov(retornos del activo, retornos del índice) / var(retornos
    del índice), sobre ~1 año de cierres diarios. Índice de referencia: ^MERV
    para tickers .BA (mercado argentino), SPY para el resto (ADRs y US)."""
    try:
        bench_symbol = '^MERV' if symbol.upper().endswith('.BA') else 'SPY'
        bench_closes = _get_benchmark_closes(bench_symbol)
        if bench_closes is None or len(bench_closes) < 30:
            return None
        df = pd.DataFrame({'a': _naive_dates(close_series), 'b': _naive_dates(bench_closes)}).dropna()
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
    TTM Squeeze Momentum — LazyBear. Versión vectorizada (pandas rolling): mismo
    resultado que la anterior pero sin loops de Python por vela; la regresión
    lineal solo se calcula en las 2 últimas velas, que son las únicas que se usan.
    Devuelve: { sqzOn, sqzOff, sqzMom, sqzMomPrev }
    """
    try:
        if len(hist) < length + 5:
            return None
        close = hist['Close'].astype(float)
        high  = hist['High'].astype(float)
        low   = hist['Low'].astype(float)

        basis    = close.rolling(length).mean()
        dev      = close.rolling(length).std(ddof=0)
        bb_upper = basis + mult_bb * dev
        bb_lower = basis - mult_bb * dev

        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_kc   = tr.rolling(length).mean()
        kc_upper = basis + mult_kc * atr_kc
        kc_lower = basis - mult_kc * atr_kc

        sqz_on  = bool(((bb_lower > kc_lower) & (bb_upper < kc_upper)).iloc[-1])
        sqz_off = bool(((bb_lower < kc_lower) & (bb_upper > kc_upper)).iloc[-1])

        mid   = (high.rolling(length).max() + low.rolling(length).min()) / 2
        delta = (close - (mid + basis) / 2).values
        x  = np.arange(length, dtype=float)
        xm = x.mean()
        denom = float(np.sum((x - xm) ** 2))

        def mom_at(idx):
            if idx < length - 1:
                return float('nan')
            y = delta[idx - length + 1: idx + 1]
            if len(y) < length or np.isnan(y).any():
                return float('nan')
            ym = y.mean()
            slope = float(np.sum((x - xm) * (y - ym)) / denom) if denom else 0.0
            return slope * (length - 1) + (ym - slope * xm)

        n = len(delta)
        m_last, m_prev = mom_at(n - 1), mom_at(n - 2)
        if np.isnan(m_last):
            return None
        return {
            'sqzOn':      sqz_on,
            'sqzOff':     sqz_off,
            'sqzMom':     round(float(m_last), 4),
            'sqzMomPrev': round(float(m_prev), 4) if not np.isnan(m_prev) else None,
        }
    except Exception:
        return None


# ─── SCORE DE FUERZA RELATIVA (estilo IBD) + VOLUMEN COMPRA/VENTA ─────────────

def calc_rs_score(closes):
    """Fuerza relativa de precio estilo IBD: 40% del retorno del último trimestre +
    20% de cada uno de los 3 trimestres anteriores (los últimos 3 meses pesan el doble).
    Es la base del ranking 'L' (Leader) de CANSLIM. Si hay menos de 12 meses de
    historial, usa los trimestres disponibles y reescala los pesos."""
    n = len(closes)
    num = den = 0.0
    for i, w in enumerate((0.4, 0.2, 0.2, 0.2)):
        a, b = 63 * i, 63 * (i + 1)
        if n <= b:
            break
        p_new, p_old = closes[-1 - a], closes[-1 - b]
        if p_old and p_old > 0:
            num += w * (p_new / p_old - 1)
            den += w
    if den < 0.4:
        return None
    return round(num / den * 100, 2)


def calc_updown_ratio(closes, volumes, n=50):
    """Volumen acumulado en ruedas de suba / en ruedas de baja (últimas n ruedas).
    >1 = acumulación (más volumen cuando sube que cuando baja). O'Neil usa 50 ruedas."""
    try:
        n = min(n, len(closes) - 1)
        if n < 10:
            return None
        up = down = 0.0
        for i in range(len(closes) - n, len(closes)):
            if closes[i] > closes[i - 1]:
                up += volumes[i]
            elif closes[i] < closes[i - 1]:
                down += volumes[i]
        if down <= 0:
            return None
        return round(up / down, 2)
    except Exception:
        return None


# ─── TÉCNICOS: todo lo que sale del historial diario (sin llamar a info/financials) ───

def _tech_from_hist(symbol, hist):
    hist = hist.dropna(subset=['Close'])
    if hist.empty or len(hist) < 5:
        return None
    close_series = hist['Close'].astype(float)
    closes  = close_series.values.tolist()
    volumes = hist['Volume'].fillna(0).astype(float).values.tolist()

    price      = round(closes[-1], 4)
    prev_close = round(closes[-2], 4) if len(closes) >= 2 else price
    chg_pct    = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0

    ema50, ema200 = calc_ema(closes, 50), calc_ema(closes, 200)
    mom3, mom6 = momentum(closes, 63), momentum(closes, 126)
    mom12 = momentum(closes, 252) if len(closes) >= 253 else mom6

    vol_today = int(volumes[-1]) if volumes else 0
    avg_vol20 = int(np.mean(volumes[-20:])) if len(volumes) >= 20 else vol_today
    vol_week, avg_vol_week = calc_weekly_volume(hist)

    highs = hist['High'].replace(0, np.nan).astype(float)
    lows  = hist['Low'].replace(0, np.nan).astype(float)
    high52 = float(highs.max()) if highs.notna().any() else float(max(closes))
    low52  = float(lows.min())  if lows.notna().any()  else float(min(closes))
    day_high = float(highs.iloc[-1]) if pd.notna(highs.iloc[-1]) else None
    day_low  = float(lows.iloc[-1])  if pd.notna(lows.iloc[-1])  else None
    high52, low52 = round(high52, 4), round(low52, 4)

    sqz = calc_squeeze_momentum(hist) or {}

    return {
        'symbol':        symbol,
        'shortName':     symbol,
        'currency':      'ARS' if symbol.endswith('.BA') else 'USD',
        'price':         price,
        'prevClose':     prev_close,
        'chgPct':        chg_pct,
        'rsi':           calc_rsi(closes),
        'adx':           calc_adx(hist),
        'ema50':         ema50,
        'ema200':        ema200,
        'sma50':         calc_sma(closes, 50),
        'sma200':        calc_sma(closes, 200),
        'distEma50':     pct_dist(price, ema50),
        'distEma200':    pct_dist(price, ema200),
        'mom3m':         mom3,
        'mom6m':         mom6,
        'mom12m':        mom12,
        'rsScore':       calc_rs_score(closes),
        'volume':        vol_today,
        'avgVol20':      avg_vol20,
        'volWeek':       vol_week,
        'avgVolWeek':    avg_vol_week,
        'volAnnualized': calc_volatility_annualized(closes, period=20),
        # Máx/Mín reales (intradiarios) → el frontend ya no necesita los proxies CORS
        'high52':        high52,
        'low52':         low52,
        'yearHigh':      high52,
        'yearLow':       low52,
        'dayHigh':       round(day_high, 4) if day_high is not None else None,
        'dayLow':        round(day_low, 4) if day_low is not None else None,
        'distFromHigh':  pct_dist(price, high52),
        'distFromLow':   pct_dist(price, low52),
        'betaCalc':      calc_beta_local(close_series, symbol),
        'upDownVolRatio': calc_updown_ratio(closes, volumes, 50),   # 50 ruedas (O'Neil)
        'upDownVol20':    calc_updown_ratio(closes, volumes, 20),
        'baseDepth':      calc_base_depth(hist),
        'daysSinceHigh':  calc_days_since_high(hist),
        'spark':         [round(c, 2) for c in closes[-20:]],
        'sqzOn':         sqz.get('sqzOn'),
        'sqzOff':        sqz.get('sqzOff'),
        'sqzMom':        sqz.get('sqzMom'),
        'sqzMomPrev':    sqz.get('sqzMomPrev'),
    }


def calc_base_depth(hist, window=75):
    """Profundidad (%) de la base actual: caída desde el máximo de las últimas ~15 semanas hasta el
    mínimo posterior. O'Neil: bases sanas rara vez superan 33-35%; más profundas suelen fallar."""
    try:
        h = hist['High'].astype(float).values[-window:]
        l = hist['Low'].astype(float).values[-window:]
        i = int(np.nanargmax(h))
        peak = h[i]
        if not peak or peak <= 0:
            return None
        trough = float(np.nanmin(l[i:]))
        return round((peak - trough) / peak * 100, 1)
    except Exception:
        return None


def calc_days_since_high(hist):
    """Ruedas transcurridas desde el máximo de 52 semanas (0 = hizo máximo hoy)."""
    try:
        h = hist['High'].astype(float).values
        return int(len(h) - 1 - int(np.nanargmax(h)))
    except Exception:
        return None


def _fetch_tech(symbol):
    hist = yf.Ticker(symbol).history(period='1y', interval='1d', auto_adjust=True)
    if hist is None or hist.empty:
        return None
    return _tech_from_hist(symbol, hist)


def _pick_symbol_frame(df, sym, n_syms):
    """Saca el DataFrame de UN símbolo de un yf.download (maneja las distintas
    formas de columnas que devuelven las versiones de yfinance)."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df if n_syms == 1 else None
    lv0 = df.columns.get_level_values(0)
    lv1 = df.columns.get_level_values(1)
    if sym in lv0:
        return df[sym]
    if sym in lv1:
        return df.xs(sym, axis=1, level=1)
    return None


def _download_many(symbols, period='1y', interval='1d', auto_adjust=True):
    """UNA sola llamada a Yahoo para muchos símbolos (mucho más rápido que N llamadas
    sueltas). Devuelve {symbol: DataFrame OHLCV}; los que no vengan quedan afuera."""
    out = {}
    if not symbols:
        return out
    try:
        df = yf.download(tickers=symbols, period=period, interval=interval, auto_adjust=auto_adjust,
                         group_by='ticker', threads=True, progress=False)
    except Exception:
        traceback.print_exc()
        return out
    if df is None or df.empty:
        return out
    for s in symbols:
        try:
            sub = _pick_symbol_frame(df, s, len(symbols))
            if sub is None:
                continue
            sub = sub.dropna(how='all')
            if 'Close' in sub.columns and sub['Close'].notna().sum() >= 5:
                out[s] = sub
        except Exception:
            continue
    return out


@app.route('/techbatch', methods=['POST'])
def get_tech_batch():
    """Indicadores técnicos de MUCHOS tickers en un solo request (y una sola
    descarga a Yahoo). Es la carga rápida del scanner: precio, RSI, ADX, EMAs,
    volumen, momentum, Máx/Mín 52s, fuerza relativa... Los fundamentals de CANSLIM
    (EPS, ROE, institucional) llegan aparte por /fundamentals, cacheados."""
    data = request.get_json(silent=True) or {}
    symbols = []
    for s in (data.get('symbols') or []):
        s = str(s or '').upper().strip()
        if s and re.match(r'^[A-Z0-9.^=\-]{1,14}$', s) and s not in symbols:
            symbols.append(s)
    symbols = symbols[:80]
    results = {}
    missing = []
    for s in symbols:
        c = cache_get('tech:' + s, TECH_TTL)
        if c is not None:
            results[s] = c
        else:
            missing.append(s)

    if missing:
        frames = _download_many(missing)
        for s, hist in frames.items():
            try:
                t = _tech_from_hist(s, hist)
                if t:
                    cache_set('tech:' + s, t)
                    results[s] = t
            except Exception:
                traceback.print_exc()
        # Los que la descarga masiva no trajo: pedido individual en paralelo (red de seguridad)
        still = [s for s in missing if s not in results]
        if still:
            with ThreadPoolExecutor(max_workers=min(8, len(still))) as ex:
                for s, t in zip(still, ex.map(lambda x: cached_call('tech:' + x, TECH_TTL, lambda: _fetch_tech(x)), still)):
                    if t:
                        results[s] = t
    out = {s: (results.get(s) or {'symbol': s, 'error': 'No data'}) for s in symbols}
    return jsonify({'results': out})


# ─── FUNDAMENTALS (CANSLIM: C, A, I, S) — cacheados horas ─────────────────────

def _row_series(df, names):
    """Fila de un estado financiero como Serie ordenada de más reciente a más vieja."""
    if df is None or getattr(df, 'empty', True):
        return None, None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            try:
                s = s.astype(float)
            except Exception:
                continue
            if len(s):
                s.index = pd.to_datetime(s.index)
                return s.sort_index(ascending=False), n
    return None, None


def _yoy_growths(s):
    """Crecimiento interanual de cada trimestre vs. el trimestre de ~1 año atrás,
    emparejado por FECHA (no por posición: si falta un trimestre en Yahoo, el
    emparejamiento por posición compara contra el trimestre equivocado).
    Base <= 0 → no hay % con sentido: devuelve None para ese trimestre."""
    out = []
    dates = list(s.index)
    vals = list(s.values)
    for i, d in enumerate(dates):
        g = None
        for j in range(i + 1, len(dates)):
            gap = (d - dates[j]).days
            if 320 <= gap <= 410:
                base = vals[j]
                if base > 0:
                    g = round((vals[i] - base) / base * 100, 1)
                break
        out.append(g)
    return out


_INFL_CACHE_KEY = 'ar_infl'


def _ar_inflation():
    """Inflación argentina (%): interanual (últimos 12 meses) y CAGR de 3 años (últimos 36 meses), compuestas
    desde la serie mensual de api.argentinadatos.com. Sirve para medir el crecimiento de balances en pesos en
    términos reales. Si la API falla: AR_INFLATION_YOY (por defecto 30) y AR_INFLATION_CAGR3 (por defecto 60)."""
    def build():
        arr = _http_json('https://api.argentinadatos.com/v1/finanzas/indices/inflacion', timeout=15)
        vals = [_num(x.get('valor')) for x in arr if isinstance(x, dict)]
        vals = [v for v in vals if v is not None]
        if len(vals) < 12:
            return None

        def comp(vs):
            acc = 1.0
            for v in vs:
                acc *= (1 + v / 100.0)
            return acc
        yoy = (comp(vals[-12:]) - 1) * 100
        cagr3 = ((comp(vals[-36:]) ** (12.0 / min(36, len(vals)))) - 1) * 100 if len(vals) >= 24 else None
        return {'yoy': round(yoy, 1), 'cagr3': round(cagr3, 1) if cagr3 is not None else None}
    try:
        v = cached_call(_INFL_CACHE_KEY, 12 * 3600, build)
        if v:
            return v
    except Exception:
        pass
    try:
        return {'yoy': float(os.environ.get('AR_INFLATION_YOY', '30')),
                'cagr3': float(os.environ.get('AR_INFLATION_CAGR3', '60'))}
    except Exception:
        return {'yoy': 30.0, 'cagr3': 60.0}


def _ar_inflation_yoy():
    return _ar_inflation()['yoy']


def _fetch_fund(symbol):
    t = yf.Ticker(symbol)
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_info = ex.submit(lambda: t.info or {})
        f_q    = ex.submit(lambda: t.quarterly_financials)
        f_a    = ex.submit(lambda: t.financials)
        try:    info = f_info.result(timeout=25)
        except Exception: info = {}
        try:    qf = f_q.result(timeout=25)
        except Exception: qf = None
        try:    af = f_a.result(timeout=25)
        except Exception: af = None

    def safe(key, default=None):
        v = info.get(key, default)
        return v if v not in (None, 'None', '', 'N/A') else default

    # C — EPS por acción (Diluted EPS) en vez de Net Income: el EPS descuenta recompras
    # y dilución, que es lo que CANSLIM mira. Si Yahoo no lo trae, cae a Net Income.
    eps_q, eps_src = _row_series(qf, ['Diluted EPS', 'Basic EPS', 'Net Income'])
    rev_q, _ = _row_series(qf, ['Total Revenue', 'Operating Revenue'])
    eps_growth_q = eps_growth_prev = eps_accel = None
    if eps_q is not None:
        g = _yoy_growths(eps_q)
        eps_growth_q    = g[0] if len(g) > 0 else None
        eps_growth_prev = g[1] if len(g) > 1 else None
        if eps_growth_q is not None and eps_growth_prev is not None:
            eps_accel = round(eps_growth_q - eps_growth_prev, 1)
    rev_growth_q = None
    if rev_q is not None:
        g = _yoy_growths(rev_q)
        rev_growth_q = g[0] if g else None

    # A — crecimiento anual: último año + CAGR a 3 años (O'Neil pide ~25% sostenido, no un año suelto)
    eps_a, _ = _row_series(af, ['Diluted EPS', 'Basic EPS', 'Net Income'])
    eps_growth_a = eps_cagr3 = None
    eps_stable = None      # True = ningún año de caída en el EPS (O'Neil pide crecimiento estable, no un año suelto)
    eps_years = 0
    if eps_a is not None and len(eps_a) >= 2:
        v = list(eps_a.values)
        _gs = [(v[i] - v[i + 1]) / abs(v[i + 1]) for i in range(min(3, len(v) - 1)) if v[i + 1] != 0]
        eps_years = len(_gs)
        if _gs:
            eps_stable = all(g >= 0 for g in _gs)
        if v[1] > 0:
            eps_growth_a = round((v[0] - v[1]) / v[1] * 100, 1)
        if len(v) >= 4 and v[3] > 0 and v[0] > 0:
            eps_cagr3 = round(((v[0] / v[3]) ** (1 / 3) - 1) * 100, 1)

    def pct(key, nd=1):
        x = safe(key)
        try:
            return round(float(x) * 100, nd) if x is not None else None
        except Exception:
            return None

    pe = safe('trailingPE') or safe('forwardPE')
    beta = safe('beta')
    div_rate = safe('dividendRate')
    px_ref = safe('currentPrice') or safe('regularMarketPrice') or safe('previousClose')
    div_y = round(float(div_rate) / float(px_ref) * 100, 2) if (div_rate and px_ref) else None
    return {
        'fundTs':        int(time.time()),
        'shortName':     safe('shortName', symbol),
        'currency':      safe('currency', 'ARS' if symbol.endswith('.BA') else 'USD'),
        'marketCap':     safe('marketCap'),
        'pe':            round(pe, 2) if pe else None,
        'beta':          round(beta, 3) if beta else None,
        'divYield':      div_y,
        'epsGrowthQ':    eps_growth_q,
        'epsGrowthQPrev': eps_growth_prev,
        'epsAccel':      eps_accel,
        'epsGrowthA':    eps_growth_a,
        'epsCagr3':      eps_cagr3,
        'epsSource':     'EPS' if (eps_src and 'EPS' in eps_src) else ('Net Income' if eps_src else None),
        'epsStable':     eps_stable,
        'epsYears':      eps_years or None,
        'inflYoY':       _ar_inflation()['yoy'] if symbol.endswith('.BA') else None,
        'inflCagr3':     _ar_inflation()['cagr3'] if symbol.endswith('.BA') else None,
        'fundV':         2,          # versión del esquema: el frontend re-pide fundamentals viejos sin estos campos
        'revGrowthQ':    rev_growth_q,
        'roe':           pct('returnOnEquity'),
        'profitMargin':  pct('profitMargins'),
        'floatShares':   safe('floatShares'),
        'sharesOutstanding': safe('sharesOutstanding'),
        'heldPctInstitutions': pct('heldPercentInstitutions'),
    }


@app.route('/fundamentals', methods=['GET'])
def get_fundamentals():
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    f = cached_call('fund:' + symbol, FUND_TTL, lambda: _fetch_fund(symbol))
    if not f:
        return jsonify({'error': 'No fundamentals for ' + symbol}), 404
    out = dict(f)
    out['symbol'] = symbol
    return jsonify(out)


@app.route('/quote', methods=['GET'])
def get_quote():
    """Cotización + indicadores de UN ticker. ?fast=1 → solo técnicos (sin fundamentals).
    Sin fast, técnicos y fundamentals se piden EN PARALELO y ambos salen del cache si están frescos."""
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400
    fast = request.args.get('fast') == '1'
    try:
        if fast:
            tech = cached_call('tech:' + symbol, TECH_TTL, lambda: _fetch_tech(symbol))
            fund = None
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                ft = ex.submit(cached_call, 'tech:' + symbol, TECH_TTL, lambda: _fetch_tech(symbol))
                ff = ex.submit(cached_call, 'fund:' + symbol, FUND_TTL, lambda: _fetch_fund(symbol))
                tech, fund = ft.result(), ff.result()
        if not tech:
            return jsonify({'error': 'No data for ' + symbol}), 404
        result = dict(tech)
        if fund:
            result.update(fund)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── MERCADO: dirección (M de CANSLIM), VIX, macro ────────────────────────────

MARKET_INDEXES = [('SPY', 'S&P 500'), ('QQQ', 'Nasdaq 100'), ('IWM', 'Russell 2000'), ('^MERV', 'Merval')]
MARKET_MACRO   = [('^VIX', 'VIX'), ('^TNX', 'Bono EEUU 10A (%)'), ('DX-Y.NYB', 'Dólar DXY'),
                  ('GC=F', 'Oro'), ('CL=F', 'Petróleo WTI'), ('BTC-USD', 'Bitcoin')]


def _dist_days(hist, lookback=25):
    """Días de distribución (O'Neil): ruedas con caída >= 0,2% y volumen mayor al de
    la rueda anterior, en las últimas 25. 5 o más = presión vendedora institucional."""
    try:
        c = hist['Close'].astype(float).values
        v = hist['Volume'].fillna(0).astype(float).values
        if v[-lookback:].sum() <= 0:
            return None   # índices sin volumen (ej. Merval en Yahoo)
        n = 0
        for i in range(max(1, len(c) - lookback), len(c)):
            if c[i] / c[i - 1] - 1 <= -0.002 and v[i] > v[i - 1]:
                n += 1
        return n
    except Exception:
        return None


def _index_state(sym, name, hist):
    closes = hist['Close'].astype(float).dropna().values.tolist()
    if len(closes) < 30:
        return None
    price, prev = closes[-1], closes[-2]
    ema50, ema200 = calc_ema(closes, 50), calc_ema(closes, 200)
    d50, d200 = pct_dist(price, ema50), pct_dist(price, ema200)
    dd = _dist_days(hist)
    if d200 is not None and d50 is not None:
        if d50 > 0 and d200 > 0:
            status = 'alcista'
        elif d200 > 0:
            status = 'presión'
        elif d50 > 0:
            status = 'recuperación'
        else:
            status = 'bajista'
        if status == 'alcista' and dd is not None and dd >= 5:
            status = 'presión'
    else:
        status = None
    return {
        'symbol': sym, 'name': name, 'price': round(price, 2),
        'chgPct': round((price / prev - 1) * 100, 2),
        'mom1m': momentum(closes, 21), 'mom3m': momentum(closes, 63),
        'rsScore': calc_rs_score(closes),
        'distEma50': d50, 'distEma200': d200, 'distDays': dd, 'status': status,
        'ok': status == 'alcista',
    }


def _build_market():
    syms = [s for s, _ in MARKET_INDEXES] + [s for s, _ in MARKET_MACRO]
    frames = _download_many(syms, period='1y')
    idx, macro = [], []
    for s, name in MARKET_INDEXES:
        if s in frames:
            st = _index_state(s, name, frames[s])
            if st:
                idx.append(st)
    for s, name in MARKET_MACRO:
        if s in frames:
            cl = frames[s]['Close'].astype(float).dropna().values.tolist()
            if len(cl) >= 2:
                macro.append({'symbol': s, 'name': name, 'price': round(cl[-1], 2),
                              'chgPct': round((cl[-1] / cl[-2] - 1) * 100, 2),
                              'mom1m': momentum(cl, 21)})
    if not idx and not macro:
        return None
    by = {i['symbol']: i for i in idx}
    def m_for(sym, label):
        s = by.get(sym)
        if not s:
            return None
        tip = '%s: %s | vs EMA50 %s%% | vs EMA200 %s%%' % (
            label, s['status'] or 's/d',
            ('%+.1f' % s['distEma50']) if s['distEma50'] is not None else '--',
            ('%+.1f' % s['distEma200']) if s['distEma200'] is not None else '--')
        if s['distDays'] is not None:
            tip += ' | %d días de distribución (25 ruedas)' % s['distDays']
        return {'ok': s['ok'], 'tip': tip}
    return {'asOf': datetime.utcnow().isoformat() + 'Z', 'indexes': idx, 'macro': macro,
            'm': {'us': m_for('SPY', 'S&P 500 (SPY)'), 'ar': m_for('^MERV', 'Merval')}}


def get_market_data():
    return cached_call('market', MARKET_TTL, _build_market)


@app.route('/market', methods=['GET'])
def market_route():
    d = get_market_data()
    if not d:
        return jsonify({'error': 'No se pudo obtener el mercado'}), 502
    return jsonify(d)


# ─── FEAR & GREED (CNN) con respaldo propio ───────────────────────────────────

CNN_FG_URL = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata'
FG_INDICATORS = {
    'market_momentum_sp500': 'Momentum del S&P 500 (vs media 125d)',
    'stock_price_strength':  'Fortaleza de precios (nuevos máx vs mín 52s)',
    'stock_price_breadth':   'Amplitud (volumen de subas vs bajas)',
    'put_call_options':      'Opciones put/call',
    'market_volatility_vix': 'Volatilidad (VIX vs media 50d)',
    'junk_bond_demand':      'Demanda de bonos basura',
    'safe_haven_demand':     'Refugio seguro (bonos vs acciones)',
}

def _fg_label(score):
    if score is None: return None
    if score < 25: return 'Miedo extremo'
    if score < 45: return 'Miedo'
    if score < 55: return 'Neutral'
    if score < 75: return 'Codicia'
    return 'Codicia extrema'

def _fetch_cnn_fg():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Origin': 'https://edition.cnn.com',
        'Referer': 'https://edition.cnn.com/',
    }
    r = requests.get(CNN_FG_URL, headers=headers, timeout=8)
    r.raise_for_status()
    j = r.json()
    fg = j.get('fear_and_greed') or {}
    score = fg.get('score')
    if score is None:
        return None
    def f1(x):
        try: return round(float(x), 1)
        except Exception: return None
    inds = []
    for key, label in FG_INDICATORS.items():
        o = j.get(key) or {}
        sc = f1(o.get('score'))
        if sc is not None:
            inds.append({'key': key, 'label': label, 'score': sc, 'rating': _fg_label(sc)})
    pts = (j.get('fear_and_greed_historical') or {}).get('data') or []
    hist = [f1(p.get('y')) for p in pts[-130:] if p.get('y') is not None]
    return {
        'source': 'CNN', 'estimated': False,
        'score': f1(score), 'rating': _fg_label(float(score)),
        'previousClose': f1(fg.get('previous_close')), 'week': f1(fg.get('previous_1_week')),
        'month': f1(fg.get('previous_1_month')), 'year': f1(fg.get('previous_1_year')),
        'indicators': inds, 'history': hist,
        'asOf': fg.get('timestamp'),
    }

def _estimate_fg():
    """Respaldo si CNN no responde: índice ESTIMADO con VIX (50%) + momentum del SPY vs su
    media de 125 ruedas (50%). No es el índice de CNN: se marca como estimado."""
    frames = _download_many(['SPY', '^VIX'], period='1y')
    spy, vix = frames.get('SPY'), frames.get('^VIX')
    if spy is None or vix is None:
        return None
    sc = spy['Close'].astype(float).dropna()
    vc = vix['Close'].astype(float).dropna()
    if len(sc) < 130 or len(vc) < 5:
        return None
    mom = (sc.iloc[-1] / sc.rolling(125).mean().iloc[-1] - 1) * 100
    s_mom = max(0.0, min(100.0, (mom + 8) / 16 * 100))
    s_vix = max(0.0, min(100.0, (35 - float(vc.iloc[-1])) / (35 - 12) * 100))
    score = round((s_mom + s_vix) / 2, 1)
    return {
        'source': 'Estimado (VIX + momentum SPY)', 'estimated': True,
        'score': score, 'rating': _fg_label(score),
        'previousClose': None, 'week': None, 'month': None, 'year': None,
        'indicators': [
            {'key': 'vix', 'label': 'VIX actual: %.1f' % float(vc.iloc[-1]), 'score': round(s_vix, 1), 'rating': _fg_label(s_vix)},
            {'key': 'mom', 'label': 'SPY vs media 125d: %+.1f%%' % mom, 'score': round(s_mom, 1), 'rating': _fg_label(s_mom)},
        ],
        'history': [], 'asOf': datetime.utcnow().isoformat() + 'Z',
    }

def _build_fg():
    try:
        d = _fetch_cnn_fg()
        if d:
            return d
    except Exception as e:
        print('[Fear&Greed] CNN falló:', e)
    return _estimate_fg()

def get_fear_greed():
    return cached_call('feargreed', FG_TTL, _build_fg)

@app.route('/feargreed', methods=['GET'])
def feargreed_route():
    d = get_fear_greed()
    if not d:
        return jsonify({'error': 'No se pudo obtener el Fear & Greed'}), 502
    return jsonify(d)


# ─── ROTACIÓN SECTORIAL (RRG) con ETFs sectoriales SPDR vs SPY ────────────────
# Aproximación al Relative Rotation Graph (JdK) de Bloomberg/StockCharts, NO idéntica:
#   RS        = precio del sector / precio de SPY
#   RS-Ratio  = 100 + z-score de RS contra su propia media móvil (fuerza relativa normalizada)
#   RS-Mom    = 100 + cambio de RS-Ratio en k períodos, normalizado por su desvío
# >100 en ambos ejes = Líder; Ratio>100 y Mom<100 = Debilitándose; ambos <100 = Rezagado;
# Ratio<100 y Mom>100 = Mejorando. Sirve para ver HACIA DÓNDE rota el dinero entre sectores.

SECTOR_ETFS = [('XLK', 'Tecnología'), ('XLF', 'Financiero'), ('XLE', 'Energía'), ('XLV', 'Salud'),
               ('XLI', 'Industriales'), ('XLY', 'Cons. Discrecional'), ('XLP', 'Cons. Básico'),
               ('XLU', 'Utilities'), ('XLB', 'Materiales'), ('XLRE', 'Inmobiliario'), ('XLC', 'Comunicación')]

def _quadrant(ratio, mom):
    if ratio >= 100 and mom >= 100: return 'Líder'
    if ratio >= 100:                return 'Debilitándose'
    if mom >= 100:                  return 'Mejorando'
    return 'Rezagado'

def _build_rrg(period, trail):
    syms = ['SPY'] + [s for s, _ in SECTOR_ETFS]
    frames = _download_many(syms, period='2y')
    if 'SPY' not in frames:
        return None
    closes = pd.DataFrame({s: f['Close'].astype(float) for s, f in frames.items()})
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize() if getattr(closes.index, 'tz', None) is not None else pd.to_datetime(closes.index).normalize()
    closes = closes.dropna(subset=['SPY']).ffill()
    if period == 'weekly':
        closes = closes.resample('W-FRI').last().dropna(subset=['SPY'])
        w, k = 26, 4
    else:
        w, k = 60, 5
    bench = closes['SPY']
    sectors = []
    for sym, name in SECTOR_ETFS:
        if sym not in closes.columns:
            continue
        c = closes[sym]
        rs = 100 * c / bench
        ratio = 100 + (rs - rs.rolling(w).mean()) / rs.rolling(w).std(ddof=0)
        roc = ratio.diff(k)
        mom = 100 + roc / roc.rolling(w).std(ddof=0)
        df = pd.DataFrame({'x': ratio, 'y': mom}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 2:
            continue
        tail = df.tail(trail)
        cl = c.dropna()
        def ret(n):
            return round((cl.iloc[-1] / cl.iloc[-1 - n] - 1) * 100, 2) if len(cl) > n else None
        x, y = float(df['x'].iloc[-1]), float(df['y'].iloc[-1])
        sectors.append({
            'symbol': sym, 'name': name, 'ratio': round(x, 2), 'momentum': round(y, 2),
            'quadrant': _quadrant(x, y),
            'trail': [{'x': round(float(a), 2), 'y': round(float(b), 2),
                       'd': d.strftime('%Y-%m-%d')} for d, a, b in zip(tail.index, tail['x'], tail['y'])],
            'chg1d': ret(1), 'chg1w': ret(5), 'chg1m': ret(21), 'chg3m': ret(63),
        })
    if not sectors:
        return None
    return {'period': period, 'benchmark': 'SPY', 'asOf': closes.index[-1].strftime('%Y-%m-%d'),
            'sectors': sectors, 'params': {'window': w, 'rocPeriod': k, 'trail': trail}}

def get_rrg(period='daily', trail=8):
    period = 'weekly' if period == 'weekly' else 'daily'
    trail = max(3, min(16, int(trail)))
    return cached_call('rrg:%s:%d' % (period, trail), RRG_TTL, lambda: _build_rrg(period, trail))

@app.route('/rrg', methods=['GET'])
def rrg_route():
    try:
        trail = int(request.args.get('trail', 8))
    except ValueError:
        trail = 8
    mk_ = request.args.get('market', 'us')
    if mk_ == 'ar':
        d = get_rrg_ar(request.args.get('period', 'daily'), trail)
    elif mk_ == 'idx':
        d = get_rrg_idx(request.args.get('period', 'daily'), trail)
    else:
        d = get_rrg(request.args.get('period', 'daily'), trail)
    if not d:
        return jsonify({'error': 'No se pudo calcular la rotación'}), 502
    return jsonify(d)


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
        def _calc1h():
            h = yf.Ticker(symbol).history(period='10d', interval='60m', auto_adjust=True)
            if h.empty or len(h) < 25:
                return None
            return calc_squeeze_momentum(h) or None
        sqz = cached_call('sqz1h:' + symbol, 300, _calc1h)
        if not sqz:
            return jsonify({'error': 'No data'}), 404
        return jsonify({'symbol': symbol, 'sqzOn': sqz.get('sqzOn'), 'sqzOff': sqz.get('sqzOff'),
                        'sqzMom': sqz.get('sqzMom'), 'sqzMomPrev': sqz.get('sqzMomPrev')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/squeeze1hbatch', methods=['POST'])
def get_squeeze_1h_batch():
    """Squeeze en velas de 1h para MUCHOS tickers con una sola descarga a Yahoo
    (antes: un request y un pedido a Yahoo por ticker)."""
    data = request.get_json(silent=True) or {}
    symbols = []
    for s in (data.get('symbols') or []):
        s = str(s or '').upper().strip()
        if s and re.match(r'^[A-Z0-9.^=\-]{1,14}$', s) and s not in symbols:
            symbols.append(s)
    symbols = symbols[:80]
    out, missing = {}, []
    for s in symbols:
        c = cache_get('sqz1h:' + s, 300)
        if c:
            out[s] = dict(c, symbol=s)
        else:
            missing.append(s)
    if missing:
        for s, h in _download_many(missing, period='10d', interval='60m').items():
            try:
                if len(h) >= 25:
                    r = calc_squeeze_momentum(h)
                    if r:
                        cache_set('sqz1h:' + s, r)
                        out[s] = dict(r, symbol=s)
            except Exception:
                pass
    return jsonify({'results': out})


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
        # Reusa el historial técnico ya cacheado si existe (evita otro pedido a Yahoo)
        t = cache_get('tech:' + symbol, TECH_TTL)
        if t and t.get('sqzMom') is not None:
            return jsonify({'symbol': symbol, 'sqzOn': t.get('sqzOn'), 'sqzOff': t.get('sqzOff'),
                            'sqzMom': t.get('sqzMom'), 'sqzMomPrev': t.get('sqzMomPrev')})
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
    """Recibe lista de symbols y devuelve todos (en paralelo, con cache)."""
    data = request.get_json(silent=True) or {}
    symbols = [str(s).upper().strip() for s in (data.get('symbols') or []) if s][:60]
    if not symbols:
        return jsonify([])
    def one(sym):
        try:
            with app.test_request_context('/quote?symbol=' + sym):
                r = get_quote()
            resp = r[0] if isinstance(r, tuple) else r
            return json.loads(resp.get_data(as_text=True))
        except Exception as e:
            return {'symbol': sym, 'error': str(e)}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return jsonify(list(ex.map(one, symbols)))


@app.route('/preciobatch', methods=['POST'])
def get_precio_batch():
    """Auto-refresh de precios: precio, cambio % y volumen de MUCHOS tickers con una sola
    descarga (5 días) a Yahoo, en vez de un request por ticker."""
    data = request.get_json(silent=True) or {}
    symbols = []
    for s in (data.get('symbols') or []):
        s = str(s or '').upper().strip()
        if s and re.match(r'^[A-Z0-9.^=\-]{1,14}$', s) and s not in symbols:
            symbols.append(s)
    symbols = symbols[:80]
    out, missing = {}, []
    for s in symbols:
        c = cache_get('precio:' + s, PRECIO_TTL)
        if c:
            out[s] = c
        else:
            missing.append(s)
    if missing:
        for s, h in _download_many(missing, period='5d').items():
            try:
                cl = h['Close'].dropna().astype(float).values.tolist()
                if len(cl) < 2:
                    continue
                vol = h['Volume'].fillna(0).values.tolist()
                r = {'symbol': s, 'price': round(cl[-1], 4), 'prevClose': round(cl[-2], 4),
                     'chgPct': round((cl[-1] - cl[-2]) / cl[-2] * 100, 2) if cl[-2] else 0,
                     'volume': int(vol[-1]) if vol else 0}
                cache_set('precio:' + s, r)
                out[s] = r
            except Exception:
                pass
    return jsonify({'results': out})


@app.route('/precio', methods=['GET'])
def get_precio():
    """Fetch rápido: solo precio actual, cambio% y volumen del día.
    No descarga historial de 1 año — usado por el auto-refresh."""
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400
    try:
        cp = cache_get('precio:' + symbol, PRECIO_TTL)
        if cp:
            return jsonify(cp)
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
        out = {
            'symbol':    symbol,
            'price':     price,
            'prevClose': prev_close,
            'chgPct':    chg_pct,
            'volume':    volume,
        }
        cache_set('precio:' + symbol, out)
        return jsonify(out)
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

    # Panorama de mercado (cacheado) en paralelo con la pre-carga de tickers
    _mex = ThreadPoolExecutor(max_workers=1)
    _mfut = _mex.submit(market_snapshot_text)

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

    try:
        mercado_txt = _mfut.result(timeout=8)
    except Exception:
        mercado_txt = ''
    _mex.shutdown(wait=False)

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
        "4. Herramienta mercado_hoy: sentimiento (Fear & Greed), dirección de índices, VIX, macro, rotación sectorial "
        "(EEUU y Merval), rotación entre países/índices (EEUU, Brasil, México, Argentina, Europa, Japón, China, India, "
        "Corea...), dólar (oficial/MEP/CCL/blue y brechas) y riesgo país al instante. Ya viene un resumen abajo en "
        "'Estado del mercado'; llamala solo si necesitás el detalle.\n"
        "4b. Herramienta noticias_mercado: titulares recientes del mercado argentino o de Wall Street (prensa financiera). "
        "Las noticias de cada empresa dentro de datos_ticker ya combinan Yahoo, Google News y prensa argentina.\n"
        "5. Tu conocimiento general, solo para contexto (qué hace la empresa, cómo leer un indicador). Nunca para "
        "cifras actuales.\n"
        "Para ahorrar tiempo: si los datos pre-cargados ya responden la pregunta, NO uses web_search. Los campos "
        "scanner_tecnico y scanner_canslim son los mismos números que el usuario ve en su scanner (CANSLIM: C = EPS "
        "trimestral YoY con aceleración y ventas creciendo, A = EPS anual: CAGR 3 años, sin años de caída y ROE >= 17%, "
        "N = a menos de 10% del máximo de 52s con una base de profundidad sana, S = volumen compra/venta de 50 ruedas, "
        "L = fuerza relativa (percentil >= 80 en su universo y mejor que el índice), I = tenencia institucional, "
        "M = dirección del mercado; en acciones argentinas los crecimientos se miden descontando la inflación).\n\n"
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
        "Estado del mercado ahora:\n" + (mercado_txt or "(no disponible; usá mercado_hoy)") + "\n\n" +
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



# ═══════════════════════════════════════════════════════════════════════════
#  v2 — DÓLAR/RIESGO PAÍS, EARNINGS, NOTICIAS, OPCIONES, SHORT INTEREST,
#       HISTORIAL DE PRECIOS (CARTERA) Y ROTACIÓN SECTORIAL DEL MERVAL
# ═══════════════════════════════════════════════════════════════════════════
DOLAR_TTL = int(os.environ.get('DOLAR_TTL', '60'))        # seg — dólares (MEP/CCL/oficial/blue)
RIESGO_TTL = int(os.environ.get('RIESGO_TTL', '600'))     # seg — riesgo país
EARN_TTL  = int(os.environ.get('EARN_TTL', str(6 * 3600)))  # seg — fechas/EPS esperado de earnings
NEWS_TTL  = int(os.environ.get('NEWS_TTL', '600'))        # seg — noticias por ticker
OPT_TTL   = int(os.environ.get('OPT_TTL', '300'))         # seg — cadena de opciones
SI_TTL    = int(os.environ.get('SI_TTL', str(12 * 3600))) # seg — short interest (Yahoo lo actualiza 2 veces/mes)
HIST_TTL  = int(os.environ.get('HIST_TTL', str(6 * 3600)))  # seg — cierres mensuales para la cartera

_HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; DUBZ-Monitor/2.0)'}
_SYM_RE = re.compile(r'^[A-Z0-9.^=\-]{1,14}$')


def _http_json(url, timeout=8):
    r = requests.get(url, timeout=timeout, headers=_HTTP_HEADERS)
    r.raise_for_status()
    return r.json()


def _clean_symbols(raw, limit):
    out = []
    for s in (raw or []):
        s = str(s or '').upper().strip()
        if s and _SYM_RE.match(s) and s not in out:
            out.append(s)
    return out[:limit]


def _first_price(node):
    """Busca el primer campo 'price' numérico dentro de un JSON anidado (para CriptoYa)."""
    if isinstance(node, dict):
        if 'price' in node:
            return _num(node.get('price'))
        for v in node.values():
            p = _first_price(v)
            if p:
                return p
    return None


def _build_dolar():
    """Cotizaciones del dólar. Fuente principal: dolarapi.com (agrega ámbito y otras casas).
    Respaldo: CriptoYa (AL30). El frontend nunca pega contra estas APIs: lo hace el server,
    así no hay problemas de CORS ni de límites por IP del navegador."""
    out = {'asOf': datetime.utcnow().isoformat() + 'Z', 'casas': {}, 'fuentes': []}
    cas = out['casas']
    try:
        for d in _http_json('https://dolarapi.com/v1/dolares'):
            casa = d.get('casa')
            if casa:
                cas[casa] = {'compra': _num(d.get('compra')), 'venta': _num(d.get('venta')),
                             'fecha': d.get('fechaActualizacion')}
        out['fuentes'].append('dolarapi.com')
    except Exception as e:
        print('[dolar] dolarapi falló:', e)

    def venta(casa):
        c = cas.get(casa) or {}
        return c.get('venta') or c.get('compra')

    if not venta('bolsa') or not venta('contadoconliqui'):
        try:
            cy = _http_json('https://criptoya.com/api/dolar')
            if not venta('bolsa'):
                p = _first_price((cy.get('mep') or {}).get('al30')) or _first_price(cy.get('mep'))
                if p:
                    cas['bolsa'] = {'compra': p, 'venta': p, 'fecha': None}
            if not venta('contadoconliqui'):
                p = _first_price((cy.get('ccl') or {}).get('al30')) or _first_price(cy.get('ccl'))
                if p:
                    cas['contadoconliqui'] = {'compra': p, 'venta': p, 'fecha': None}
            out['fuentes'].append('criptoya.com')
        except Exception as e:
            print('[dolar] criptoya falló:', e)

    mep, ccl, ofi, blue = venta('bolsa'), venta('contadoconliqui'), venta('oficial'), venta('blue')
    if not mep and not ccl and not ofi:
        return None
    out['mep'], out['ccl'], out['oficial'], out['blue'] = mep, ccl, ofi, blue
    out['mayorista'] = venta('mayorista')

    def brecha(a, b):
        return round((a / b - 1) * 100, 2) if (a and b) else None
    out['brechaCclMep'] = brecha(ccl, mep)
    out['brechaMepOficial'] = brecha(mep, ofi)
    out['brechaCclOficial'] = brecha(ccl, ofi)
    out['brechaBlueOficial'] = brecha(blue, ofi)
    return out


def _dolar_hist(casa):
    """Serie diaria [(fecha, venta)] de ArgentinaDatos (se cachea 1 hora)."""
    def f():
        data = _http_json('https://api.argentinadatos.com/v1/cotizaciones/dolares/' + casa, timeout=15)
        pts = []
        for d in data:
            v = _num(d.get('venta') or d.get('compra'))
            if v and d.get('fecha'):
                pts.append((str(d['fecha'])[:10], v))
        pts.sort()
        return pts or None
    return cached_call('dolarhist:' + casa, 3600, f) or []


def get_dolar():
    d = cached_call('dolar', DOLAR_TTL, _build_dolar)
    return d


def _build_riesgo():
    try:
        rp = _http_json('https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais', timeout=15)
    except Exception as e:
        print('[riesgo] serie falló:', e)
        rp = None
    if isinstance(rp, list) and len(rp) >= 2:
        pts = [(str(x.get('fecha'))[:10], _num(x.get('valor'))) for x in rp if x.get('valor') is not None]
        pts = [p for p in pts if p[1] is not None]
        if len(pts) >= 2:
            last, prev = pts[-1], pts[-2]
            m1 = pts[-22] if len(pts) > 22 else pts[0]
            return {'valor': last[1], 'fecha': last[0],
                    'varDia': round(last[1] - prev[1], 1),
                    'var1m': round(last[1] - m1[1], 1),
                    'serie': [p[1] for p in pts[-90:]]}
    try:
        u = _http_json('https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo')
        return {'valor': _num(u.get('valor')), 'fecha': str(u.get('fecha'))[:10],
                'varDia': None, 'var1m': None, 'serie': []}
    except Exception as e:
        print('[riesgo] ultimo falló:', e)
        return None


@app.route('/dolar', methods=['GET'])
def dolar_route():
    d = get_dolar()
    if not d:
        return jsonify({'error': 'No se pudo obtener el dólar'}), 502
    out = dict(d)
    out['riesgoPais'] = cached_call('riesgo', RIESGO_TTL, _build_riesgo)
    # serie de brecha CCL/MEP (últimos 90 días hábiles) para el mini-gráfico
    try:
        b = dict(_dolar_hist('bolsa'))
        c = dict(_dolar_hist('contadoconliqui'))
        fechas = sorted(set(b) & set(c))[-90:]
        out['serieBrechaCclMep'] = [round((c[f] / b[f] - 1) * 100, 2) for f in fechas]
    except Exception:
        out['serieBrechaCclMep'] = []
    return jsonify(out)


@app.route('/mep-historial', methods=['GET'])
def mep_historial_route():
    """MEP de fin de cada mes: {'2025-03': 1195.5, ...}. Lo usa la Cartera para valuar en
    USD/ARS con el tipo de cambio de cada mes (y no con el MEP de hoy)."""
    pts = _dolar_hist('bolsa')
    if not pts:
        return jsonify({'error': 'sin datos'}), 502
    mm = {}
    for f, v in pts:
        mm[f[:7]] = v          # al estar ordenado por fecha, queda el último valor de cada mes
    return jsonify({'meses': mm})


# ─── EARNINGS ────────────────────────────────────────────────────────────────

def _to_iso_date(x):
    try:
        return str(x)[:10]
    except Exception:
        return None


def _fetch_earn_one(sym):
    res = {'symbol': sym, 'date': None, 'dateEnd': None, 'estimated': False,
           'epsEst': None, 'revEst': None}
    today = datetime.utcnow().date().isoformat()
    try:
        t = yf.Ticker(sym)
    except Exception:
        return res
    try:
        cal = t.calendar
        if isinstance(cal, dict) and cal:
            fe = cal.get('Earnings Date')
            lst = list(fe) if isinstance(fe, (list, tuple)) else ([fe] if fe else [])
            ds = sorted(d for d in (_to_iso_date(x) for x in lst) if d and d >= today)
            if ds:
                res['date'] = ds[0]
                if len(ds) > 1 and ds[-1] != ds[0]:   # Yahoo da un rango cuando la fecha no está confirmada
                    res['dateEnd'] = ds[-1]
                    res['estimated'] = True
            res['epsEst'] = _num(cal.get('Earnings Average'))
            res['revEst'] = _num(cal.get('Revenue Average'))
    except Exception as e:
        print('[earn] calendar', sym, e)
    if not res['date']:
        try:
            ed = t.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                fut = []
                for idx, row in ed.iterrows():
                    d = _to_iso_date(idx)
                    if d and d >= today:
                        fut.append((d, _num(row.get('EPS Estimate'))))
                if fut:
                    fut.sort()
                    res['date'], res['epsEst'] = fut[0][0], fut[0][1]
                    res['estimated'] = True
        except Exception as e:
            print('[earn] dates', sym, e)
    return res


@app.route('/earnings', methods=['POST'])
def earnings_route():
    data = request.get_json(silent=True) or {}
    symbols = _clean_symbols(data.get('symbols'), 60)
    try:
        days = max(1, min(120, int(data.get('days', 90))))
    except (TypeError, ValueError):
        days = 90
    if not symbols:
        return jsonify({'results': []})

    def one(s):
        return cached_call('earn:' + s, EARN_TTL, lambda: _fetch_earn_one(s))
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for r in ex.map(one, symbols) if r]
    today = datetime.utcnow().date()
    limit = (today + timedelta(days=days)).isoformat()
    rows = [r for r in rows if r.get('date') and today.isoformat() <= r['date'] <= limit]
    rows.sort(key=lambda r: (r['date'], r['symbol']))
    return jsonify({'results': rows, 'asOf': datetime.utcnow().isoformat() + 'Z'})


# ─── NOTICIAS (multi-fuente) ─────────────────────────────────────────────────
# Antes salían SOLO de yfinance, que casi no trae noticias de acciones argentinas (.BA) ni de
# papeles chicos: por eso "Noticias de mi cartera" quedaba vacío. Ahora cada ticker junta:
#   1) yfinance            (Yahoo, el que ya estaba)
#   2) Yahoo Finance RSS   (feed de titulares por ticker; a veces trae lo que yfinance no)
#   3) Google News RSS     (por nombre de empresa; en español/Argentina para los .BA)
#   4) Prensa argentina    (Ámbito, El Cronista, Infobae, iProfesional vía Google News, solo .BA)
# Si aun así quedan pocas, se agregan titulares generales del mercado (símbolo 'MERCADO').
NEWS_PER_SYMBOL = int(os.environ.get('NEWS_PER_SYMBOL', '6'))
NEWS_MAX_AGE_DAYS = int(os.environ.get('NEWS_MAX_AGE_DAYS', '21'))

AR_NEWS_NAMES = {
    'GGAL': 'Grupo Financiero Galicia', 'YPFD': 'YPF', 'YPF': 'YPF', 'PAMP': 'Pampa Energía', 'PAM': 'Pampa Energía',
    'BMA': 'Banco Macro', 'SUPV': 'Grupo Supervielle', 'BBAR': 'BBVA Argentina', 'BBVA': 'BBVA Argentina',
    'BYMA': 'BYMA Bolsas y Mercados Argentinos', 'VIST': 'Vista Energy', 'TGSU2': 'Transportadora de Gas del Sur',
    'TGS': 'Transportadora de Gas del Sur', 'TGNO4': 'Transportadora de Gas del Norte', 'CEPU': 'Central Puerto',
    'EDN': 'Edenor', 'TRAN': 'Transener', 'TXAR': 'Ternium Argentina', 'ALUA': 'Aluar', 'LOMA': 'Loma Negra',
    'CRES': 'Cresud', 'MIRG': 'Mirgor', 'COME': 'Sociedad Comercial del Plata', 'TECO2': 'Telecom Argentina',
    'TEO': 'Telecom Argentina', 'IRS': 'IRSA', 'IRSA': 'IRSA', 'METR': 'Metrogas', 'HARG': 'Holcim Argentina',
    'VALO': 'Grupo Financiero Valores', 'CVH': 'Cablevisión Holding', 'AGRO': 'Agrometal', 'BHIP': 'Banco Hipotecario',
    'BPAT': 'Banco Patagonia', 'CAPX': 'Capex', 'DGCU2': 'Distribuidora de Gas Cuyana', 'GCLA': 'Grupo Clarín',
    'LEDE': 'Ledesma', 'MOLI': 'Molinos Río de la Plata', 'OEST': 'Oeste Grupo Concesionario', 'PGR': 'Phoenix Global Resources',
    'RICH': 'Laboratorios Richmond', 'SEMI': 'Molinos Semino', 'GBAN': 'Gas Natural BAN', 'BOLT': 'Boldt',
    'MELI': 'MercadoLibre', 'GLOB': 'Globant', 'DESP': 'Despegar', 'BIOX': 'Bioceres', 'CRESY': 'Cresud',
}
AR_PRESS_SITES = '(site:ambito.com OR site:cronista.com OR site:infobae.com OR site:iprofesional.com OR site:lanacion.com.ar)'


def _to_iso_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat() + 'Z'


def _rss_items(url, sym, limit=10, timeout=8, default_source=None):
    """Lee un feed RSS y devuelve [{symbol,title,source,ts,url}]. Nunca levanta excepción."""
    out = []
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0 (compatible; DUBZ-Monitor/2.0)'})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for it in root.iter('item'):
            title = _html.unescape((it.findtext('title') or '').strip())
            link = (it.findtext('link') or '').strip()
            if not title or not link:
                continue
            se = it.find('source')
            source = (se.text or '').strip() if (se is not None and se.text) else default_source
            # Google News agrega " - Medio" al final del título: lo sacamos y queda en 'source'
            if source and title.endswith(' - ' + source):
                title = title[:-(len(source) + 3)].strip()
            elif se is None and 'news.google.com' in url and ' - ' in title:
                head, _, tail = title.rpartition(' - ')
                if head and 2 < len(tail) < 40:
                    title, source = head.strip(), tail.strip()
            ts = None
            pub = it.findtext('pubDate')
            if pub:
                try:
                    ts = _to_iso_utc(parsedate_to_datetime(pub))
                except Exception:
                    ts = None
            out.append({'symbol': sym, 'title': title, 'source': source, 'ts': ts, 'url': link})
            if len(out) >= limit:
                break
    except Exception as e:
        print('[news-rss]', url[:80], e)
    return out


def _gnews_url(query, ar):
    q = quote_plus(query + ' when:14d')
    if ar:
        return 'https://news.google.com/rss/search?q=%s&hl=es-419&gl=AR&ceid=AR:es-419' % q
    return 'https://news.google.com/rss/search?q=%s&hl=en-US&gl=US&ceid=US:en' % q


def _company_name(sym):
    base = sym[:-3] if sym.endswith('.BA') else sym
    if base in AR_NEWS_NAMES:
        return AR_NEWS_NAMES[base]
    f = cache_get_stale('fund:' + sym)          # si ya se cargaron los fundamentals, tenemos el nombre
    if isinstance(f, dict) and f.get('shortName') and f.get('shortName') != sym:
        return re.sub(r'\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|S\.A\.?|Co\.?|Holdings?|Class [A-C])$', '', str(f['shortName']), flags=re.I).strip()
    return None


def _yf_news(sym):
    items = []
    try:
        for n in (yf.Ticker(sym).news or [])[:10]:
            c = n.get('content') if isinstance(n.get('content'), dict) else None
            if c:
                url = ((c.get('canonicalUrl') or {}).get('url')) or ((c.get('clickThroughUrl') or {}).get('url'))
                items.append({'symbol': sym, 'title': c.get('title'),
                              'source': (c.get('provider') or {}).get('displayName'),
                              'ts': c.get('pubDate'), 'url': url})
            else:
                ts = n.get('providerPublishTime')
                items.append({'symbol': sym, 'title': n.get('title'), 'source': n.get('publisher'),
                              'ts': datetime.utcfromtimestamp(ts).isoformat() + 'Z' if ts else None,
                              'url': n.get('link')})
    except Exception as e:
        print('[news-yf]', sym, e)
    return [i for i in items if i.get('title') and i.get('url')]


def _norm_title(t):
    return re.sub(r'[^a-z0-9áéíóúñ]+', ' ', (t or '').lower()).strip()[:90]


def _news_items(sym):
    """Noticias de un ticker juntando todas las fuentes en paralelo (cada una falla sola)."""
    ar = sym.endswith('.BA')
    base = sym[:-3] if ar else sym
    name = _company_name(sym)
    jobs = [lambda: _yf_news(sym),
            lambda: _rss_items('https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US' % quote_plus(sym),
                               sym, 10, 8, 'Yahoo Finance')]
    if ar:
        q1 = ('"%s" acciones OR bolsa OR balance OR resultados' % name) if name else ('%s acciones bolsa' % base)
        q2 = ('%s %s' % (name or base, AR_PRESS_SITES))
        jobs.append(lambda: _rss_items(_gnews_url(q1, True), sym, 10, 8))
        jobs.append(lambda: _rss_items(_gnews_url(q2, True), sym, 10, 8))
    else:
        q = ('"%s" stock' % name) if name else ('%s stock' % base)
        jobs.append(lambda: _rss_items(_gnews_url(q, False), sym, 10, 8))
    lists = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(j) for j in jobs]
        for f in futs:
            try:
                lists.append(f.result(timeout=14) or [])
            except Exception:
                lists.append([])
    cutoff = (datetime.utcnow() - timedelta(days=NEWS_MAX_AGE_DAYS)).isoformat()
    seen, merged = set(), []
    for lst in lists:
        for it in lst:
            k = _norm_title(it.get('title'))
            if not k or k in seen:
                continue
            if it.get('ts') and str(it['ts'])[:19] < cutoff:      # muy viejas: afuera
                continue
            seen.add(k)
            merged.append(it)
    merged.sort(key=lambda i: i.get('ts') or '', reverse=True)
    return merged[:12]          # [] si no hay nada (cached_call lo cachea y no insiste en cada request)


def _market_news(ar=True, limit=8):
    """Titulares generales del mercado (para cuando las acciones de la cartera no tienen noticias propias)."""
    if ar:
        qs = [('Merval acciones bolsa argentina', True), ('mercado argentino bonos riesgo país dólar %s' % AR_PRESS_SITES, True)]
    else:
        qs = [('Wall Street stocks S&P 500 Nasdaq', False)]
    items = []
    for q, a_ in qs:
        items += _rss_items(_gnews_url(q, a_), 'MERCADO', limit, 8)
    seen, out = set(), []
    for it in sorted(items, key=lambda i: i.get('ts') or '', reverse=True):
        k = _norm_title(it.get('title'))
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out[:limit]


@app.route('/news', methods=['POST'])
def news_route():
    data = request.get_json(silent=True) or {}
    symbols = _clean_symbols(data.get('symbols'), 30)
    try:
        limit = max(1, min(80, int(data.get('limit', 30))))
    except (TypeError, ValueError):
        limit = 30
    if not symbols:
        return jsonify({'results': []})

    def one(s):
        return cached_call('news:' + s, NEWS_TTL, lambda: _news_items(s)) or []
    with ThreadPoolExecutor(max_workers=8) as ex:
        lists = list(ex.map(one, symbols))
    # tope por ticker para que uno solo no tape a los demás; después se ordena todo por fecha
    seen, merged, per_sym = set(), [], {}
    for lst in lists:
        for it in lst:
            k = _norm_title(it.get('title'))
            s = it.get('symbol')
            if not k or k in seen or per_sym.get(s, 0) >= NEWS_PER_SYMBOL:
                continue
            seen.add(k)
            per_sym[s] = per_sym.get(s, 0) + 1
            merged.append(it)
    covered = sorted({i['symbol'] for i in merged})
    # Si hay pocas noticias propias, sumar titulares generales del mercado (símbolo 'MERCADO')
    general = False
    if len(merged) < 8:
        has_ar = any(s.endswith('.BA') for s in symbols)
        extra = cached_call('news:mkt:%s' % ('ar' if has_ar else 'us'), NEWS_TTL, lambda: _market_news(has_ar)) or []
        for it in extra:
            k = _norm_title(it.get('title'))
            if k and k not in seen:
                seen.add(k)
                merged.append(it)
                general = True
    merged.sort(key=lambda i: i.get('ts') or '', reverse=True)
    return jsonify({'results': merged[:limit], 'asOf': datetime.utcnow().isoformat() + 'Z',
                    'covered': covered, 'missing': [s for s in symbols if s not in covered], 'general': general})


@app.route('/news/mercado', methods=['GET'])
def news_mercado_route():
    ar = request.args.get('market', 'ar') != 'us'
    items = cached_call('news:mkt:%s' % ('ar' if ar else 'us'), NEWS_TTL, lambda: _market_news(ar, 12)) or []
    return jsonify({'results': items})


# ─── OPCIONES: PUT/CALL Y FLUJO INUSUAL ──────────────────────────────────────

def _f(v, default=0.0):
    try:
        x = float(v)
        return default if x != x else x
    except Exception:
        return default


def _build_options(sym, max_days=45, max_exp=4):
    t = yf.Ticker(sym)
    exps = list(t.options or [])
    if not exps:
        return None
    today = datetime.utcnow().date()
    sel = []
    for e in exps:
        try:
            dte = (datetime.strptime(e, '%Y-%m-%d').date() - today).days
        except Exception:
            continue
        if dte < 0:
            continue
        if dte <= max_days or not sel:
            sel.append((e, dte))
        if len(sel) >= max_exp:
            break
    if not sel:
        return None
    price = None
    try:
        price = _num(t.fast_info.get('last_price'))
    except Exception:
        pass
    if not price:
        try:
            h = t.history(period='5d')
            price = _num(h['Close'].dropna().iloc[-1])
        except Exception:
            price = None

    cv = pv = coi = poi = 0.0
    by_exp, unusual, ivs = [], [], []
    for e, dte in sel:
        try:
            ch = t.option_chain(e)
        except Exception as ex_:
            print('[options] chain', sym, e, ex_)
            continue
        e_cv = e_pv = e_coi = e_poi = 0.0
        for side, df in (('C', ch.calls), ('P', ch.puts)):
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                vol, oi, last = _f(r.get('volume')), _f(r.get('openInterest')), _f(r.get('lastPrice'))
                strike = _f(r.get('strike'))
                if side == 'C':
                    e_cv += vol; e_coi += oi
                else:
                    e_pv += vol; e_poi += oi
                prem = vol * last * 100
                is_unusual = vol >= 500 and prem >= 50000 and ((oi > 0 and vol / oi >= 2) or (oi == 0 and vol >= 1000))
                if is_unusual:
                    unusual.append({'exp': e, 'dte': dte, 'strike': strike, 'side': side,
                                    'vol': int(vol), 'oi': int(oi),
                                    'volOi': round(vol / oi, 1) if oi > 0 else None,
                                    'last': round(last, 2), 'premium': int(prem),
                                    'itm': bool(r.get('inTheMoney')),
                                    'iv': _num(r.get('impliedVolatility'))})
                if price and dte <= 30 and abs(strike - price) / price <= 0.03:
                    iv = _f(r.get('impliedVolatility'), None)
                    if iv:
                        ivs.append(iv)
        cv += e_cv; pv += e_pv; coi += e_coi; poi += e_poi
        by_exp.append({'exp': e, 'dte': dte, 'callVol': int(e_cv), 'putVol': int(e_pv),
                       'pcVol': round(e_pv / e_cv, 2) if e_cv else None,
                       'pcOi': round(e_poi / e_coi, 2) if e_coi else None})
    if not by_exp:
        return None
    unusual.sort(key=lambda u: u['premium'], reverse=True)
    pc_vol = round(pv / cv, 2) if cv else None
    if pc_vol is None:
        sentiment = None
    elif pc_vol < 0.7:
        sentiment = 'alcista'
    elif pc_vol > 1.0:
        sentiment = 'bajista'
    else:
        sentiment = 'neutral'
    return {'symbol': sym, 'price': price, 'asOf': datetime.utcnow().isoformat() + 'Z',
            'expirations': [e for e, _ in sel],
            'callVol': int(cv), 'putVol': int(pv), 'callOi': int(coi), 'putOi': int(poi),
            'pcVol': pc_vol, 'pcOi': round(poi / coi, 2) if coi else None,
            'ivAtm': round(sum(ivs) / len(ivs) * 100, 1) if ivs else None,
            'sentiment': sentiment, 'byExp': by_exp, 'unusual': unusual[:12]}


@app.route('/options', methods=['GET'])
def options_route():
    sym = request.args.get('symbol', '').upper().strip()
    if not sym or not _SYM_RE.match(sym):
        return jsonify({'error': 'símbolo inválido'}), 400
    d = cached_call('opt:' + sym, OPT_TTL, lambda: _build_options(sym))
    if not d:
        return jsonify({'error': 'Sin opciones listadas para ' + sym}), 404
    return jsonify(d)


# ─── SHORT INTEREST ──────────────────────────────────────────────────────────

def _fetch_si(sym):
    info = yf.Ticker(sym).info or {}
    spf = info.get('shortPercentOfFloat')
    ss = info.get('sharesShort')
    if spf is None and ss is None:
        return None
    prior = info.get('sharesShortPriorMonth')
    chg = round((ss / prior - 1) * 100, 1) if (ss and prior) else None
    dt = info.get('dateShortInterest')
    try:
        dstr = datetime.utcfromtimestamp(dt).strftime('%Y-%m-%d') if dt else None
    except Exception:
        dstr = None
    return {'symbol': sym, 'name': info.get('shortName'),
            'pctFloat': round(float(spf) * 100, 2) if spf is not None else None,
            'daysToCover': _num(info.get('shortRatio')),
            'sharesShort': int(ss) if ss else None,
            'chgMonthPct': chg, 'asOfDate': dstr}


@app.route('/shortint', methods=['POST'])
def shortint_route():
    data = request.get_json(silent=True) or {}
    symbols = _clean_symbols(data.get('symbols'), 40)
    if not symbols:
        return jsonify({'results': []})

    def one(s):
        return cached_call('si:' + s, SI_TTL, lambda: _fetch_si(s))
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for r in ex.map(one, symbols) if r]
    return jsonify({'results': rows})


# ─── HISTORIAL MENSUAL (evolución de la cartera) ─────────────────────────────

@app.route('/history', methods=['POST'])
def history_route():
    """Cierres de fin de mes SIN ajustar por dividendos (el valor real del precio ese día)."""
    data = request.get_json(silent=True) or {}
    symbols = _clean_symbols(data.get('symbols'), 40)
    period = data.get('period') if data.get('period') in ('1y', '2y', '5y', '10y', 'max') else '5y'
    if not symbols:
        return jsonify({'results': {}})

    def build():
        out = {}
        for s, h in _download_many(symbols, period=period, interval='1mo', auto_adjust=False).items():
            try:
                ser = h['Close'].dropna().astype(float)
                m = {}
                for d, v in ser.items():
                    m[d.strftime('%Y-%m')] = round(float(v), 4)
                if m:
                    out[s] = m
            except Exception:
                continue
        return out or None
    key = 'hist:%s:%s' % (period, ','.join(sorted(symbols)))
    res = cached_call(key, HIST_TTL, build) or {}
    return jsonify({'results': res, 'period': period})


# ─── ROTACIÓN SECTORIAL DEL MERVAL (canastas equiponderadas vs ^MERV) ────────

AR_SECTORS = [
    ('BANC', 'Bancos y Finanzas',       ['GGAL.BA', 'BMA.BA', 'SUPV.BA', 'BBAR.BA', 'BYMA.BA']),
    ('PYG',  'Petróleo y Gas',          ['YPFD.BA', 'VIST.BA', 'TGSU2.BA', 'TGNO4.BA']),
    ('ELEC', 'Energía eléctrica',       ['PAMP.BA', 'CEPU.BA', 'EDN.BA', 'TRAN.BA']),
    ('MAT',  'Materiales e Industria',  ['TXAR.BA', 'ALUA.BA', 'LOMA.BA']),
    ('CONS', 'Consumo, Agro y Otros',   ['CRES.BA', 'MIRG.BA', 'COME.BA', 'TECO2.BA']),
]


def _build_rrg_ar(period, trail):
    bench = '^MERV'
    syms = [bench] + sorted({m for _, _, ms in AR_SECTORS for m in ms})
    frames = _download_many(syms, period='2y')
    if bench not in frames:
        return None
    raw = pd.DataFrame({s: f['Close'].astype(float) for s, f in frames.items()})
    idx = pd.to_datetime(raw.index)
    raw.index = (idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx).normalize()
    raw = raw[~raw.index.duplicated(keep='last')].sort_index()
    raw = raw.dropna(subset=[bench]).ffill()
    rets = raw.pct_change()
    closes = pd.DataFrame({bench: raw[bench]})
    members_ok = {}
    for code, name, members in AR_SECTORS:
        cols = [m for m in members if m in rets.columns and rets[m].notna().sum() > 60]
        if not cols:
            continue
        sector_ret = rets[cols].mean(axis=1, skipna=True).fillna(0.0)
        closes[code] = 100 * (1 + sector_ret).cumprod()
        members_ok[code] = [m[:-3] for m in cols]
    if len(closes.columns) < 2:
        return None
    if period == 'weekly':
        closes = closes.resample('W-FRI').last().dropna(subset=[bench])
        w, k = 26, 4
    else:
        w, k = 60, 5
    sectors = []
    for code, name, _ in AR_SECTORS:
        if code not in closes.columns:
            continue
        c = closes[code]
        rs = 100 * c / closes[bench]
        ratio = 100 + (rs - rs.rolling(w).mean()) / rs.rolling(w).std(ddof=0)
        roc = ratio.diff(k)
        mom = 100 + roc / roc.rolling(w).std(ddof=0)
        df = pd.DataFrame({'x': ratio, 'y': mom}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 2:
            continue
        tail = df.tail(trail)
        cl = c.dropna()

        def ret(n, cl=cl):
            return round((cl.iloc[-1] / cl.iloc[-1 - n] - 1) * 100, 2) if len(cl) > n else None
        x, y = float(df['x'].iloc[-1]), float(df['y'].iloc[-1])
        sectors.append({
            'symbol': code, 'name': name, 'members': members_ok.get(code, []),
            'ratio': round(x, 2), 'momentum': round(y, 2), 'quadrant': _quadrant(x, y),
            'trail': [{'x': round(float(a), 2), 'y': round(float(b), 2), 'd': d.strftime('%Y-%m-%d')}
                      for d, a, b in zip(tail.index, tail['x'], tail['y'])],
            'chg1d': ret(1), 'chg1w': ret(5), 'chg1m': ret(21), 'chg3m': ret(63),
        })
    if not sectors:
        return None
    return {'market': 'ar', 'period': period, 'benchmark': 'MERVAL', 'asOf': closes.index[-1].strftime('%Y-%m-%d'),
            'sectors': sectors, 'params': {'window': w, 'rocPeriod': k, 'trail': trail}}


def get_rrg_ar(period='daily', trail=8):
    period = 'weekly' if period == 'weekly' else 'daily'
    trail = max(3, min(16, int(trail)))
    return cached_call('rrg:ar:%s:%d' % (period, trail), RRG_TTL, lambda: _build_rrg_ar(period, trail))



# ─── ROTACIÓN DE ÍNDICES Y PAÍSES (ETFs en dólares vs ACWI) ───────────────────
# Todos son ETFs que cotizan en USD, así los países son comparables entre sí (el ^MERV en pesos NO lo es:
# la inflación y el tipo de cambio lo distorsionan, por eso Argentina entra vía ARGT). Benchmark: ACWI
# (acciones del mundo) y no SPY, porque medir países contra EEUU solo diría "gana o pierde contra EEUU";
# SPY entra como un miembro más.
IDX_BENCH = ('ACWI', 'ACWI (mundo)')
IDX_ETFS = [
    ('SPY',  'EEUU · S&P 500'),        ('QQQ',  'EEUU · Nasdaq 100'),    ('IWM',  'EEUU · Small caps'),
    ('EFA',  'Desarrollados ex-EEUU'), ('EEM',  'Emergentes'),           ('EWZ',  'Brasil'),
    ('ARGT', 'Argentina (USD)'),       ('EWW',  'México'),               ('VGK',  'Europa'),
    ('EWU',  'Reino Unido'),           ('EWJ',  'Japón'),                ('MCHI', 'China'),
    ('INDA', 'India'),                 ('EWY',  'Corea del Sur'),        ('EWT',  'Taiwán'),
    ('EWC',  'Canadá'),
]


def _build_rrg_idx(period, trail):
    bench = IDX_BENCH[0]
    syms = [bench] + [s for s, _ in IDX_ETFS]
    frames = _download_many(syms, period='2y')
    if bench not in frames:
        return None
    raw = pd.DataFrame({s: f['Close'].astype(float) for s, f in frames.items()})
    idx = pd.to_datetime(raw.index)
    raw.index = (idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx).normalize()
    raw = raw[~raw.index.duplicated(keep='last')].sort_index()
    closes = raw.dropna(subset=[bench]).ffill()
    if period == 'weekly':
        closes = closes.resample('W-FRI').last().dropna(subset=[bench])
        w, k = 26, 4
    else:
        w, k = 60, 5
    sectors = []
    for sym, name in IDX_ETFS:
        if sym not in closes.columns or closes[sym].notna().sum() < w + k + 5:
            continue
        c = closes[sym]
        rs = 100 * c / closes[bench]
        ratio = 100 + (rs - rs.rolling(w).mean()) / rs.rolling(w).std(ddof=0)
        roc = ratio.diff(k)
        mom = 100 + roc / roc.rolling(w).std(ddof=0)
        df = pd.DataFrame({'x': ratio, 'y': mom}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < 2:
            continue
        tail = df.tail(trail)
        cl = c.dropna()

        def ret(n, cl=cl):
            return round((cl.iloc[-1] / cl.iloc[-1 - n] - 1) * 100, 2) if len(cl) > n else None
        x, y = float(df['x'].iloc[-1]), float(df['y'].iloc[-1])
        sectors.append({
            'symbol': sym, 'name': name, 'ratio': round(x, 2), 'momentum': round(y, 2), 'quadrant': _quadrant(x, y),
            'trail': [{'x': round(float(a), 2), 'y': round(float(b), 2), 'd': d.strftime('%Y-%m-%d')}
                      for d, a, b in zip(tail.index, tail['x'], tail['y'])],
            'chg1d': ret(1), 'chg1w': ret(5), 'chg1m': ret(21), 'chg3m': ret(63),
        })
    if not sectors:
        return None
    return {'market': 'idx', 'period': period, 'benchmark': IDX_BENCH[0], 'benchmarkName': IDX_BENCH[1],
            'asOf': closes.index[-1].strftime('%Y-%m-%d'), 'sectors': sectors,
            'params': {'window': w, 'rocPeriod': k, 'trail': trail}}


def get_rrg_idx(period='daily', trail=8):
    period = 'weekly' if period == 'weekly' else 'daily'
    trail = max(3, min(16, int(trail)))
    return cached_call('rrg:idx:%s:%d' % (period, trail), RRG_TTL, lambda: _build_rrg_idx(period, trail))


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
