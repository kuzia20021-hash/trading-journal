"""Режим рынка и риск-скейлинг по волатильности (по мотивам FinRL).

Turbulence index: насколько ТЕКУЩИЙ шаг цены отклонился от среднего,
на фоне собственной волатильности. Важно: тренд сам по себе высокой
turbulence не дает - линейный рост дает стабильные доходности и turb ~ 0.
Высокая turbulence = разворот/шум, а не направление движения.

Шкала нормирована по 99-му перцентилю истории, поэтому порог 0.8 означает
"худший 1% наблюдений".
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Regime:
    turbulence: float
    volatility: float
    drawdown: float
    high_vol: bool
    risk_multiplier: float
    label: str


def turbulence_series(closes: List[float], window: int = 20) -> np.ndarray:
    """Нормированная серия turbulence для всех точек истории."""
    c = np.array(closes, dtype=np.float64)
    n = len(c)
    out = np.zeros(n)
    if n < 3:
        return out
    logp = np.log(np.maximum(c, 1e-12))
    rets = np.diff(logp, prepend=logp[0])
    raw = np.zeros(n)
    for i in range(window, n):
        hist = rets[i - window: i]
        sd = float(hist.std())
        if sd < 1e-12:
            raw[i] = 0.0
        else:
            raw[i] = abs(float(rets[i]) - float(hist.mean())) / sd
    nz = raw[raw > 0]
    q = float(np.percentile(nz, 99)) if len(nz) else 1.0
    return raw / max(q, 1e-12)


def _metrics(closes: List[float], window: int = 20) -> Tuple[float, float, float]:
    """(turbulence, аннуализированная волатильность, просадка от локального максимума)."""
    if len(closes) < window + 2:
        return 0.0, 0.0, 0.0
    c = np.array(closes, dtype=np.float64)
    turb = float(turbulence_series(c, window)[-1])
    logp = np.log(np.maximum(c, 1e-12))
    rets = np.diff(logp)
    vol = float(np.std(rets[-window:]) * np.sqrt(252))
    peak = float(np.max(c))
    dd = max(0.0, (peak - float(c[-1])) / peak * 100) if peak > 0 else 0.0
    return turb, vol, dd


def turbulence_index(closes: List[float], window: int = 20) -> float:
    return _metrics(closes, window)[0]


def realized_vol(closes: List[float], window: int = 20) -> float:
    return _metrics(closes, window)[1]


def drawdown_pct(closes: List[float]) -> float:
    """Просадка текущей цены от максимума истории."""
    return _metrics(closes)[2]


def classify(turb: float, threshold: float = 0.5, scale: float = 0.5,
             drawdown: float = 0.0, dd_threshold: float = 15.0) -> Regime:
    """Высокий риск = turbulence выше порога ИЛИ глубокая просадка от максимума.

    Просадка нужна отдельным триггером: после обвала рынок может
    "успокоиться" (turbulence падает), но убыток уже накоплен.
    """
    high = turb > threshold or drawdown > dd_threshold
    reason = []
    if turb > threshold:
        reason.append("turbulence")
    if drawdown > dd_threshold:
        reason.append("просадка")
    return Regime(round(turb, 4), 0.0, round(drawdown, 2), high, scale if high else 1.0,
                  "высокий риск (" + ", ".join(reason) + ")" if high else "нормальный режим")


def risk_scale(closes: List[float], threshold: float = 0.5, scale: float = 0.5,
               dd_threshold: float = 15.0) -> Regime:
    turb, vol, dd = _metrics(closes)
    r = classify(turb, threshold, scale, dd, dd_threshold)
    r.volatility = round(vol, 4)
    return r


def suggest_size(base_margin: float, closes: List[float], threshold: float = 0.5,
                 scale: float = 0.5, dd_threshold: float = 15.0) -> dict:
    reg = risk_scale(closes, threshold, scale, dd_threshold)
    margin = base_margin * reg.risk_multiplier
    return {
        "base_margin": round(base_margin, 2),
        "recommended_margin": round(margin, 2),
        "risk_multiplier": reg.risk_multiplier,
        "turbulence": reg.turbulence,
        "annualized_volatility": reg.volatility,
        "drawdown_pct": reg.drawdown,
        "label": reg.label,
        "reason": ("риск выше нормы -> размер урезан вдвое" if reg.high_vol
                   else "режим нормальный -> размер не меняется"),
    }


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--closes", required=True)
    p.add_argument("--margin", type=float, default=25.0)
    a = p.parse_args()
    print(json.dumps(suggest_size(a.margin, [float(x) for x in a.closes.split(",")]),
                     ensure_ascii=False, indent=2))
