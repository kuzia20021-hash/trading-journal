"""Живой режим рынка: считает turbulence и просадку по реальной истории.

Источник истории: Yahoo Finance chart API (дневные закрытия, без ключей).
Фильтр режима применяется к размеру позиции: в высоком риске маржа режется вдвое.

Просадка считается по СКОЛЬЗЯЩЕМУ окну (по умолчанию 60 дней), а не от
годового максимума: на падающем рынке годовая просадка всегда огромная
и фильтр режет размер постоянно, что делает его бесполезным.
"""
import json
import urllib.request
from typing import List, Optional

import numpy as np

try:
    from regime import turbulence_series
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from regime import turbulence_series

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# тикер Yahoo -> тикер Walt
YAHOO_TO_WALT = {
    "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL", "XMR-USD": "XMR",
    "GC=F": "XAU", "SI=F": "XAG", "BABA": "BABA", "LIT": "LIT", "^GSPC": "SPY",
    "TSLA": "TSLA", "NVDA": "NVDA", "AAPL": "AAPL", "AMZN": "AMZN",
    "GOOGL": "GOOGL", "SPY": "SPY",
}


def fetch_closes(yahoo_ticker: str, rng: str = "1y") -> Optional[List[float]]:
    """Дневные закрытия из Yahoo Finance. None, если данных нет."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
           f"?range={rng}&interval=1d")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        return [x for x in closes if x is not None]
    except Exception:
        return None


def drawdown_window(closes: List[float], window: int = 60) -> float:
    """Просадка последней цены от максимума внутри окна."""
    if not closes:
        return 0.0
    seg = closes[-window:] if len(closes) >= window else closes
    peak = max(seg)
    return max(0.0, (peak - seg[-1]) / peak * 100) if peak > 0 else 0.0


def analyze(closes: List[float], window: int = 60, turb_threshold: float = 0.5,
            dd_threshold: float = 12.0, scale: float = 0.5) -> dict:
    """Полная картина режима по истории закрытий."""
    if not closes or len(closes) < 25:
        return {"error": "недостаточно истории"}
    c = np.array(closes, dtype=np.float64)
    turb = float(turbulence_series(c)[-1])
    logp = np.log(np.maximum(c, 1e-12))
    rets = np.diff(logp)
    vol = float(np.std(rets[-20:]) * np.sqrt(252))
    dd = drawdown_window(closes, window)
    high = turb > turb_threshold or dd > dd_threshold
    reasons = []
    if turb > turb_threshold:
        reasons.append("turbulence")
    if dd > dd_threshold:
        reasons.append("просадка")
    return {
        "turbulence": round(turb, 4),
        "annualized_volatility_pct": round(vol * 100, 2),
        "drawdown_pct": round(dd, 2),
        "drawdown_window": window,
        "high_risk": high,
        "risk_multiplier": scale if high else 1.0,
        "label": ("высокий риск (" + ", ".join(reasons) + ")") if high else "нормальный режим",
    }


def recommend_margin(closes: List[float], base_margin: float, **kw) -> dict:
    """Рекомендованная маржа с учётом режима."""
    a = analyze(closes, **kw)
    if "error" in a:
        return a
    a["base_margin"] = round(base_margin, 2)
    a["recommended_margin"] = round(base_margin * a["risk_multiplier"], 2)
    a["reason"] = ("риск выше нормы -> маржа урезана вдвое" if a["high_risk"]
                   else "режим нормальный -> маржа не меняется")
    return a


def walt_symbol(yahoo_ticker: str) -> Optional[str]:
    return YAHOO_TO_WALT.get(yahoo_ticker)


if __name__ == "__main__":
    import argparse, json as _json
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True, help="тикер Yahoo, напр. BTC-USD или BABA")
    p.add_argument("--margin", type=float, default=33.0)
    a = p.parse_args()
    closes = fetch_closes(a.ticker)
    print(_json.dumps(recommend_margin(closes, a.margin), ensure_ascii=False, indent=2))


# --- предторговый фильтр ---

# тикер Walt -> тикер Yahoo (для тех, кого нет в YAHOO_TO_WALT, но история есть)
EXTRA_YAHOO = {
    "ZRO": "ZRO-USD", "NEAR": "NEAR-USD", "AAVE": "AAVE-USD", "WLD": "WLD-USD",
    "HYPE": "HYPE-USD", "ZEC": "ZEC-USD", "LINK": "LINK-USD", "ADA": "ADA-USD",
    "MSTR": "MSTR", "MU": "MU", "INTC": "INTC", "SNDK": "SNDK", "AMD": "AMD",
    "AVAX": "AVAX-USD", "SUI": "SUI-USD", "DOGE": "DOGE-USD", "META": "META",
}


def yahoo_for(walt_symbol: str) -> Optional[str]:
    """Тикер Yahoo для тикера Walt."""
    for y, w in YAHOO_TO_WALT.items():
        if w == walt_symbol:
            return y
    return EXTRA_YAHOO.get(walt_symbol)


def pre_trade_check(walt_symbol: str, base_margin: float, **kw) -> dict:
    """Проверка перед входом: режим рынка -> разрешённая маржа.

    Возвращает:
      allowed_margin — маржа с учётом риск-скейлинга
      blocked       — True, если истории нет и вход не советуем
      note          — человекочитаемое пояснение
    """
    y = yahoo_for(walt_symbol)
    if not y:
        return {"symbol": walt_symbol, "allowed_margin": 0.0, "blocked": True,
                "note": f"нет исторических данных для {walt_symbol} - вход не советую",
                "regime": None}
    closes = fetch_closes(y)
    if not closes:
        return {"symbol": walt_symbol, "allowed_margin": 0.0, "blocked": True,
                "note": f"не удалось загрузить историю {y} - вход не советую",
                "regime": None}
    a = analyze(closes, **kw)
    if "error" in a:
        return {"symbol": walt_symbol, "allowed_margin": 0.0, "blocked": True,
                "note": "недостаточно истории для расчёта режима", "regime": None}
    a["symbol"] = walt_symbol
    a["yahoo_ticker"] = y
    a["allowed_margin"] = round(base_margin * a["risk_multiplier"], 2)
    a["base_margin"] = round(base_margin, 2)
    a["blocked"] = False
    a["note"] = (f"{a['label']}: маржа {a['allowed_margin']}$ из {base_margin}$ "
                 f"(turbulence {a['turbulence']}, просадка 60д {a['drawdown_pct']}%)")
    return a
