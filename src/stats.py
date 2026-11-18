"""Статистика журнала сделок: винрейт, средний R, макс. просадка."""
import csv
import sys
from dataclasses import dataclass


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


def load(path: str) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("r"):
                continue
            rows.append({"r": float(r["r"]), "side": r.get("side", ""), "reason": r.get("reason", "")})
    return rows


def compute(rows: list) -> Stat:
    if not rows:
        return Stat(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rs = [r["r"] for r in rows]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in rs:
        equity += x
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins) / len(rs) * 100
    expectancy = avg_win * win_rate / 100 + avg_loss * (1 - win_rate / 100)
    return Stat(len(rs), len(wins), len(losses), round(win_rate, 1), round(sum(rs) / len(rs), 3),
                round(sum(rs), 3), round(max(rs), 3), round(min(rs), 3), round(max_dd, 3),
                round(avg_win, 3), round(avg_loss, 3), round(expectancy, 3))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "journal/trades.csv"
    s = compute(load(path))
    print(f"Сделок: {s.trades} | Винрейт: {s.win_rate}%")
    print(f"Средний R: {s.avg_r} | Суммарный R: {s.total_r}")
    print(f"Лучшая: {s.best_r} | Худшая: {s.worst_r} | Макс. просадка: {s.max_drawdown_r} R")
    print(f"Средний win: {s.avg_win_r} | Средний loss: {s.avg_loss_r} | Матожидание: {s.expectancy} R")
