"""Бэктест без look-ahead.

Ключевой принцип (из TradingAgents): решение, принятое на закрытии бара D,
оценивается по закрытию бара ПОСЛЕ D. Никаких данных, известных только позже D.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Bar:
    date: str
    close: float
    high: float = 0.0
    low: float = 0.0
    atr: float = 0.0


@dataclass
class Decision:
    date: str
    side: str                 # long | short | flat
    stop: Optional[float] = None
    target: Optional[float] = None
    thesis: str = ""


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    side: str
    entry: float
    exit: float
    r: float
    reason: str
    thesis: str = ""


@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)

    @property
    def total_r(self) -> float:
        return sum(t.r for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.r > 0) / len(self.trades) * 100


def simulate(bars: list, decisions: list, atr_mult: float = 2.0) -> BacktestResult:
    """Один вход за раз. Решение применяется к следующему бару после даты решения."""
    by_date = {b.date: b for b in bars}
    result = BacktestResult()
    used = False

    for d in decisions:
        if used:
            break
        idx = next((i for i, b in enumerate(bars) if b.date > d.date), None)
        if idx is None or d.side == "flat":
            continue
        bar = bars[idx]
        entry = bar.close
        if d.side == "long":
            stop = d.stop if d.stop is not None else entry - bar.atr * atr_mult
            risk = entry - stop
        else:
            stop = d.stop if d.stop is not None else entry + bar.atr * atr_mult
            risk = stop - entry
        if risk <= 0:
            continue

        target = d.target
        for b in bars[idx + 1:]:
            hit_stop = b.low <= stop if d.side == "long" else b.high >= stop
            hit_target = target is not None and (b.high >= target if d.side == "long" else b.low <= target)
            if hit_stop:
                exit_price, reason = stop, "stop"
                break
            if hit_target:
                exit_price, reason = target, "target"
                break
        else:
            exit_price, reason = bars[-1].close, "end_of_data"

        r = (exit_price - entry) / risk if d.side == "long" else (entry - exit_price) / risk
        result.trades.append(Trade(d.date, bars[min(idx + 1, len(bars) - 1)].date, d.side,
                                   entry, exit_price, round(r, 3), reason, d.thesis))
        used = True
    return result


def make_signal_fn(signal: Callable[[Bar], Optional[Decision]]) -> Callable:
    """Обёртка: сигнальная функция по бару -> решение."""
    def inner(bar: Bar) -> Optional[Decision]:
        return signal(bar)
    return inner
