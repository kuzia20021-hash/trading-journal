"""Gym-стиль торговая среда (адаптация подходов FinRL).

Цикл state -> action -> reward -> next_state. Есть штраф за oversized-позицию
и режим turbulence (высокая волатильность -> риск режется).

Бухгалтерия: позиция знаковая (long > 0, short < 0).
    equity = cash + position * price
    покупка (delta > 0):  cash -= delta * price
    продажа (delta < 0): cash += |delta| * price
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from regime import turbulence_series
except ImportError:  # pragma: no cover - запуск как скрипт
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from regime import turbulence_series


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class EnvConfig:
    initial_cash: float = 1000.0
    taker_fee_pct: float = 0.09
    slippage_pct: float = 0.02
    max_position_pct: float = 0.25
    oversize_penalty: float = 0.5
    turbulence_threshold: float = 0.8
    risk_scale_in_turbulence: float = 0.5


@dataclass
class Action:
    target_weight: float = 0.0
    direction: int = 0          # +1 long, -1 short, 0 flat


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class TradingEnv:
    """Среда с одним активом, шаг = один бар."""

    def __init__(self, bars: List[Bar], config: Optional[EnvConfig] = None):
        self.bars = bars
        self.cfg = config or EnvConfig()
        self.reset()

    def _features(self, i: int) -> np.ndarray:
        b = self.bars
        closes = np.array([x.close for x in b[: i + 1]], dtype=np.float64)
        rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        vol = float(np.std(rets[-20:])) if len(rets) >= 2 else 0.0
        mom = float(closes[-1] / closes[-6] - 1.0) if len(closes) >= 6 else 0.0
        rng = float((b[i].high - b[i].low) / max(b[i].close, 1e-12))
        return np.array([b[i].close, mom, vol, rng, float(self.turbulence[i]),
                         self.position, self.cash], dtype=np.float64)

    def reset(self) -> np.ndarray:
        self.i = 0
        self.cash = float(self.cfg.initial_cash)
        self.position = 0.0
        self.equity_curve = []
        self.peak_equity = self.cash
        self.max_drawdown = 0.0
        self.trades = []
        closes = np.array([x.close for x in self.bars], dtype=np.float64)
        self.turbulence = turbulence_series(closes)
        return self._features(0)

    @property
    def equity(self) -> float:
        price = self.bars[self.i].close
        return self.cash + self.position * price

    def _cost(self, notional: float) -> float:
        return abs(notional) * (self.cfg.taker_fee_pct + self.cfg.slippage_pct) / 100

    def _apply_action(self, action: Action) -> float:
        price = self.bars[self.i].close
        equity = max(self.equity, 1e-12)
        penalty = 0.0

        direction = action.direction if action.direction in (-1, 0, 1) else 0
        target = abs(action.target_weight)
        if direction == 0:
            target = 0.0

        # режем размер в режиме высокой волатильности
        if self.turbulence[self.i] > self.cfg.turbulence_threshold:
            target *= self.cfg.risk_scale_in_turbulence

        desired_units = direction * target * equity / max(price, 1e-12)
        delta = desired_units - self.position

        if abs(delta) > 1e-12:
            notional = abs(delta) * price
            fee = self._cost(notional)
            # закрываем старую позицию, если разворачиваемся
            if self.position != 0 and (self.position > 0) != (delta > 0):
                close_notional = abs(self.position) * price
                self.cash += self.position * price
                self.trades.append({"date": self.bars[self.i].date, "side": "reverse",
                                    "price": price, "notional": close_notional})
                self.position = 0.0
                delta = desired_units
                notional = abs(delta) * price
                fee = self._cost(notional)
            self.cash -= delta * price
            self.cash -= fee
            self.position += delta
            if abs(self.position) > 1e-12:
                self.trades.append({"date": self.bars[self.i].date,
                                    "side": "long" if self.position > 0 else "short",
                                    "price": price, "notional": notional, "fee": fee})

        # штраф за превышение лимита позиции
        if self.position != 0:
            weight = abs(self.position * price) / max(equity, 1e-12)
            if weight > self.cfg.max_position_pct:
                excess = (weight - self.cfg.max_position_pct) / self.cfg.max_position_pct
                penalty = -self.cfg.oversize_penalty * min(excess, 3.0)
        return penalty

    def step(self, action: Action) -> StepResult:
        penalty = self._apply_action(action)
        prev_equity = self.equity

        self.i += 1
        done = self.i >= len(self.bars) - 1
        equity = self.equity

        ret = (equity - prev_equity) / max(prev_equity, 1e-12)
        reward = ret * 100.0 + penalty

        self.equity_curve.append(equity)
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            self.max_drawdown = max(self.max_drawdown,
                                    (self.peak_equity - equity) / self.peak_equity)
        return StepResult(state=self._features(self.i), reward=reward, done=done,
                          info={"equity": equity, "turbulence": float(self.turbulence[self.i]),
                                "position": self.position, "penalty": penalty})

    def run(self, policy) -> dict:
        state = self.reset()
        rewards = []
        while True:
            out = self.step(policy(state, self.i))
            rewards.append(out.reward)
            state = out.state
            if out.done:
                break
        return {
            "final_equity": round(self.equity, 2),
            "total_return_pct": round((self.equity / self.cfg.initial_cash - 1) * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }
