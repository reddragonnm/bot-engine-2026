"""
ICM (Independent Chip Model) for IIT Pokerbots 2026.

For heads-up play: ICM value = (your_chips / total_chips) * prize_pool
More generally for n players: uses Malmuth-Weitzman ICM.
"""

import math
from typing import Optional


class ICMCalculator:
    """
    ICM calculator for tournament equity.
    For heads-up: simple chip equity
    For multi-way: uses Malmuth-Weitzman approximation
    """

    def __init__(self, prize_pool: float = 100.0, places_paid: int = 2):
        self.prize_pool = prize_pool
        self.places_paid = places_paid

    def _malmuth_weitzman(
        self, stacks: list[int], prizes: list[float]
    ) -> list[float]:
        """
        Malmuth-Weitzman ICM approximation.
        Returns equity for each player given their stack and prizes.
        """
        n = len(stacks)
        if n == 0:
            return []

        total = sum(stacks)
        if total == 0:
            return [0.0] * n

        equity = [0.0] * n

        for i in range(n):
            si = stacks[i]
            if si <= 0:
                continue

            for r in range(1, min(n, len(prizes)) + 1):
                if r > len(prizes):
                    break

                prob_finish_r = 0.0

                remaining = total - si
                if remaining <= 0:
                    if r == 1:
                        prob_finish_r = 1.0
                else:
                    for k in range(r):
                        prob = (si / total) * math.prod(
                            [(total - si - j * remaining / (n - 1)) / (total - j * remaining / (n - 1))
                             for j in range(k)]
                        )
                        prob_finish_r += prob

                equity[i] += prob_finish_r * prizes[r - 1]

        return equity

    def heads_up_equity(self, my_chips: int, opp_chips: int) -> float:
        """Simple chip equity for heads-up."""
        total = my_chips + opp_chips
        if total == 0:
            return 0.5
        return my_chips / total

    def icm_equity(self, stacks: list[int], prizes: Optional[list[float]] = None) -> list[float]:
        """
        Calculate ICM equity for all players.
        
        Args:
            stacks: List of chip counts for each player
            prizes: List of prizes for each place (1st, 2nd, etc.)
                   If None, uses equal distribution scaled by prize_pool
        
        Returns:
            List of ICM equity values (in dollars) for each player
        """
        if prizes is None:
            n = len(stacks)
            if n == 2:
                prizes = [self.prize_pool * 0.6, self.prize_pool * 0.4]
            else:
                prizes = [self.prize_pool / n] * n

        if len(stacks) == 2:
            my_chips = stacks[0]
            opp_chips = stacks[1]
            my_eq = self.heads_up_equity(my_chips, opp_chips)
            opp_eq = 1.0 - my_eq
            return [my_eq * self.prize_pool, opp_eq * self.prize_pool]

        return self._malmuth_weitzman(stacks, prizes)

    def icm_value(self, my_chips: int, opp_chips: int) -> float:
        """Get ICM value for my stack in heads-up."""
        eq = self.heads_up_equity(my_chips, opp_chips)
        return eq * self.prize_pool


class ICMBotHelper:
    """
    Helper class to integrate ICM into bot decision-making.
    """

    def __init__(self, prize_pool: float = 100.0):
        self.icm = ICMCalculator(prize_pool=prize_pool)

    def should_push_icm(
        self,
        my_chips: int,
        opp_chips: int,
        pot: int,
        equity: float,
        is_bubble: bool = False,
    ) -> float:
        """
        Calculate ICM-adjusted EV for pushing all-in.
        
        Returns:
            ICM-adjusted expected value in chips
        
        Args:
            my_chips: My remaining chips
            opp_chips: Opponent's chips
            pot: Current pot
            equity: Our equity if called
            is_bubble: Whether we're in bubble situation
        """
        current_icm = self.icm.icm_value(my_chips, opp_chips)

        if my_chips >= opp_chips * 3:
            return equity * pot

        win_icm = self.icm.icm_value(my_chips + pot, opp_chips)
        lose_icm = self.icm.icm_value(my_chips - pot, opp_chips)

        icm_ev = equity * win_icm + (1 - equity) * lose_icm

        if is_bubble:
            icm_ev *= 1.1

        return icm_ev

    def fold_equity_needed(
        self,
        my_chips: int,
        opp_chips: int,
        pot: int,
        is_bubble: bool = False,
    ) -> float:
        """
        Calculate fold equity needed to make a bluff profitable by ICM.
        
        Returns:
            Required fold equity percentage
        """
        current_icm = self.icm.icm_value(my_chips, opp_chips)
        win_icm = self.icm.icm_value(my_chips + pot, opp_chips)

        icm_gain = win_icm - current_icm

        if icm_gain <= 0:
            return 1.0

        fold_needed = icm_gain / pot

        if is_bubble:
            fold_needed *= 0.9

        return min(fold_needed, 1.0)

    def tournament_mfq(
        self,
        my_chips: int,
        opp_chips: int,
        street: str,
        position: str,
    ) -> float:
        """
        Tournament-adjusted M factor (Mousel-Fishhook).
        M = M / (big blind + small blind + antes)
        
        Returns:
            Modified M for tournament play
        """
        big_blind = 20
        small_blind = 10
        ante = 0

        total_antes = ante * 2
        m_denom = big_blind + small_blind + total_antes

        if m_denom == 0:
            return float('inf')

        effective_stack = min(my_chips, opp_chips)
        m = effective_stack / m_denom

        multiplier = 1.0
        if street == "pre-flop":
            multiplier *= 1.0
        elif street in ("flop", "turn"):
            multiplier *= 0.85

        if position == "BB":
            multiplier *= 0.95

        if my_chips < big_blind * 10:
            multiplier *= 1.2

        return m * multiplier

    def risk_premium(
        self,
        my_chips: int,
        opp_chips: int,
    ) -> float:
        """
        Calculate tournament risk premium.
        Extra equity needed to call due to ICM pressure.
        
        Returns:
            Additional equity percentage needed
        """
        total_chips = my_chips + opp_chips
        if total_chips == 0:
            return 0.0

        my_share = my_chips / total_chips

        if my_share > 0.8:
            return 0.0
        if my_share < 0.2:
            return 0.08

        premium = 0.02 * (0.8 - my_share) / 0.6
        return min(premium, 0.08)

    def optimal_bet_size_icm(
        self,
        my_chips: int,
        opp_chips: int,
        pot: int,
        equity: float,
        street: str,
    ) -> int:
        """
        Calculate optimal bet size considering ICM.
        
        Returns:
            Recommended bet size in chips
        """
        risk_prem = self.risk_premium(my_chips, opp_chips)
        adj_equity = equity - risk_prem

        if adj_equity < 0.3:
            return 0

        if my_chips >= opp_chips * 3:
            min_bet = int(pot * 0.4)
            max_bet = int(pot * 0.75)
        elif my_chips >= opp_chips:
            min_bet = int(pot * 0.5)
            max_bet = int(pot * 0.8)
        else:
            min_bet = int(pot * 0.6)
            max_bet = int(pot * 1.0)

        if street == "river":
            if equity > 0.75:
                min_bet = int(pot * 0.7)
                max_bet = int(pot * 1.2)

        target = (min_bet + max_bet) // 2
        return min(target, my_chips)


def example_usage():
    """Demonstrate ICM usage."""
    icm = ICMCalculator(prize_pool=100.0)

    my_chips = 800
    opp_chips = 1200

    my_equity = icm.heads_up_equity(my_chips, opp_chips)
    print(f"Chip equity: {my_equity:.3f}")
    print(f"ICM value: ${icm.icm_value(my_chips, opp_chips):.2f}")

    helper = ICMBotHelper(prize_pool=100.0)

    risk_prem = helper.risk_premium(my_chips, opp_chips)
    print(f"Risk premium: {risk_prem:.3f} ({risk_prem*100:.1f}%)")

    optimal_bet = helper.optimal_bet_size_icm(
        my_chips, opp_chips, pot=200, equity=0.65, street="flop"
    )
    print(f"Optimal bet size: {optimal_bet}")


if __name__ == "__main__":
    example_usage()
