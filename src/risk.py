"""Риск-менеджмент: размер позиции от риска в R и свободной маржи.

Правило: риск в долларах задаётся долей от капитала (risk_pct).
Размер считается от того, что больше: риск-бюджет по стопу или ограничение по марже.
"""
from dataclasses import dataclass


@dataclass
class SizingResult:
    margin_usd: float
    notional_usd: float
    risk_usd: float
    stop_distance_pct: float
    r_multiple_to_target: float | None
    leverage: float
    notes: str


def atr_stop(entry: float, atr: float, atr_mult: float = 2.0) -> float:
    """Стоп на N ATR от входа (для лонга — вниз)."""
    return entry - atr * atr_mult


def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  leverage: float, max_leverage: float = 10.0,
                  fee_pct: float = 0.09, target: float | None = None) -> SizingResult:
    """Размер позиции.

    entry/stop/target — цены. Для шорта stop выше entry.
    risk_pct — процент капитала, который допустимо потерять на стопе.
    """
    if equity <= 0:
        raise ValueError("equity должен быть положительным")
    if stop <= 0 or entry <= 0:
        raise ValueError("цены должны быть положительными")

    side = "short" if stop > entry else "long"
    stop_distance = abs(entry - stop)
    stop_distance_pct = stop_distance / entry * 100
    if stop_distance == 0:
        raise ValueError("стоп не может совпадать с входом")

    risk_budget = equity * risk_pct / 100
    # Потери включаем комиссию входа и выхода.
    risk_per_unit = stop_distance + (entry + stop) * fee_pct / 100
    units_by_risk = risk_budget / risk_per_unit

    max_margin_by_leverage = equity * max_leverage
    notional_by_leverage = max_margin_by_leverage * leverage
    units_by_leverage = notional_by_leverage / entry

    units = min(units_by_risk, units_by_leverage)
    notional = units * entry
    margin = notional / leverage

    r_multiple = None
    if target:
        reward = abs(target - entry)
        r_multiple = reward / stop_distance

    notes = []
    if units_by_risk < units_by_leverage:
        notes.append("размер ограничен риск-бюджетом (R)")
    else:
        notes.append("размер ограничен плечом и капиталом")
    if fee_pct:
        notes.append(f"комиссия taker {fee_pct}% учтена в риске")

    return SizingResult(
        margin_usd=round(margin, 2),
        notional_usd=round(notional, 2),
        risk_usd=round(units * risk_per_unit, 2),
        stop_distance_pct=round(stop_distance_pct, 2),
        r_multiple_to_target=round(r_multiple, 2) if r_multiple else None,
        leverage=leverage,
        notes="; ".join(notes),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=float, required=True)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--entry", type=float, required=True)
    p.add_argument("--stop", type=float, required=True)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--leverage", type=int, default=5)
    a = p.parse_args()
    r = position_size(a.equity, a.risk_pct, a.entry, a.stop, a.leverage, target=a.target)
    print(f"Маржа: ${r.margin_usd} | Номинал: ${r.notional_usd} | Риск: ${r.risk_usd}")
    print(f"Расстояние до стопа: {r.stop_distance_pct}% | R к цели: {r.r_multiple_to_target}")
    print(r.notes)
