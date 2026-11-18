"""Статистика журнала сделок + бейзлайны (buy&hold и MVO).

Идея из FinRL: стратегию нужно сравнивать не с "отсутствием сделки",
а с наивными базовыми линиями, иначе результат выглядит убедительнее, чем он есть.
"""
import csv
import statistics
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class Stat:
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r: float
    total_r: float
    best_r: float
    worst_r: float
    max_drawdown_r: float
    avg_win_r: float
    avg_loss_r: float
    expectancy: float


@dataclass
class Baseline:
    name: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float


def load(path: str) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("r"):
                continue
            rows.append({"r": float(r["r"]), "side": r.get("side", ""),
                         "reason": r.get("reason", ""), "symbol": r.get("symbol", "")})
    return rows


def compute(rows: list) -> Stat:
    if not rows:
        return Stat(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rs = [r["r"] for r in rows]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    equity = peak = 0.0
    max_dd = 0.0
    for x in rs:
        equity += x
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins) / len(rs) * 100
    expectancy = avg_win * win_rate / 100 + avg_loss * (1 - win_rate / 100)
    return Stat(len(rs), len(wins), len(losses), round(win_rate, 1),
                round(sum(rs) / len(rs), 3), round(sum(rs), 3), round(max(rs), 3),
                round(min(rs), 3), round(max_dd, 3), round(avg_win, 3),
                round(avg_loss, 3), round(expectancy, 3))


def _drawdown(curve: List[float]) -> float:
    peak, dd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak:
            dd = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)


def buy_and_hold(closes: List[float]) -> Baseline:
    """Купил в начале, держит до конца."""
    if len(closes) < 2 or closes[0] == 0:
        return Baseline("buy&hold", 0.0, 0.0, 0.0)
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    curve = [c / closes[0] for c in closes]
    sharpe = 0.0
    if len(rets) > 1:
        sd = statistics.pstdev(rets)
        if sd > 0:
            sharpe = statistics.mean(rets) / sd * (252 ** 0.5)
    return Baseline("buy&hold", round((closes[-1] / closes[0] - 1) * 100, 2),
                    _drawdown(curve), round(sharpe, 2))


def mvo(closes: List[float], target_vol: float = 0.15, min_weight: float = 0.0) -> Baseline:
    """Средне-дисперсионная оптимизация: доля в рисковом активе обратно
    пропорциональна волатильности, остальное - кэш."""
    if len(closes) < 3:
        return Baseline("mvo", 0.0, 0.0, 0.0)
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    if sd <= 0:
        return buy_and_hold(closes)
    # аннуализированная волатильность
    ann = sd * (252 ** 0.5)
    weight = max(min_weight, min(1.0, target_vol / ann)) if ann > 0 else 1.0
    curve = [1.0]
    for r in rets:
        curve.append(curve[-1] * (1 + weight * r))
    sharpe = 0.0
    if len(rets) > 1:
        sr = statistics.mean(rets) * weight
        sdr = statistics.pstdev(rets) * weight
        if sdr > 0:
            sharpe = sr / sdr * (252 ** 0.5)
    return Baseline("mvo", round((curve[-1] - 1) * 100, 2), _drawdown(curve), round(sharpe, 2))


def compare(strategy_r: float, closes: List[float], risk_pct: float = 1.0) -> dict:
    """Сравнивает стратегию (в R) с бейзлайнами в процентах.

    R переводится в проценты капитала: r * risk_pct.
    """
    bh = buy_and_hold(closes)
    mv = mvo(closes)
    strat_pct = strategy_r * risk_pct
    return {
        "strategy_total_r": round(strategy_r, 3),
        "strategy_total_pct": round(strat_pct, 2),
        "risk_pct_per_r": risk_pct,
        "baselines": [
            {"name": bh.name, "total_return_pct": bh.total_return_pct,
             "max_drawdown_pct": bh.max_drawdown_pct, "sharpe": bh.sharpe},
            {"name": mv.name, "total_return_pct": mv.total_return_pct,
             "max_drawdown_pct": mv.max_drawdown_pct, "sharpe": mv.sharpe},
        ],
        "verdict": ("стратегия обгоняет бейзлайны" if strat_pct > max(bh.total_return_pct, mv.total_return_pct)
                    else "стратегия НЕ обгоняет бейзлайны - качество сигналов не доказано"),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "journal/trades.csv"
    s = compute(load(path))
    print(f"Сделок: {s.trades} | Винрейт: {s.win_rate}%")
    print(f"Средний R: {s.avg_r} | Суммарный R: {s.total_r}")
    print(f"Лучшая: {s.best_r} | Худшая: {s.worst_r} | Макс. просадка: {s.max_drawdown_r} R")
    print(f"Средний win: {s.avg_win_r} | Средний loss: {s.avg_loss_r} | Матожидание: {s.expectancy} R")


def by_regime(rows: list) -> dict:
    """Разбивка результатов по режиму рынка на момент входа.

    Пустые значения (сделки до включения фильтра) собираются в "н/д".
    """
    buckets: dict = {}
    for r in rows:
        key = r.get("regime") or "н/д"
        b = buckets.setdefault(key, {"trades": 0, "wins": 0, "total_r": 0.0})
        b["trades"] += 1
        b["total_r"] += r["r"]
        if r["r"] > 0:
            b["wins"] += 1
    out = {}
    for k, b in buckets.items():
        out[k] = {
            "trades": b["trades"],
            "wins": b["wins"],
            "win_rate": round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0,
            "total_r": round(b["total_r"], 3),
            "avg_r": round(b["total_r"] / b["trades"], 3) if b["trades"] else 0.0,
        }
    return out
