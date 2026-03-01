"""
ApexBot - Exploitative Sneak Peek Hold'em bot.

Focus:
- Fast, cached equity estimation (preflop and postflop)
- Time-aware simulation counts
- Opponent modeling (fold/aggression/bet sizing)
- Auction bidding tuned to information value and opponent patterns
- Solid, risk-aware bet sizing
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random
import math


RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def fast_hand_rank(cards: list[str]) -> float:
    if len(cards) != 2:
        return 0.5
    r1 = RANK_VALUES.get(cards[0][0], 0)
    r2 = RANK_VALUES.get(cards[1][0], 0)
    suited = cards[0][1] == cards[1][1]
    high, low = max(r1, r2), min(r1, r2)

    if r1 == r2:
        return 0.50 + (high - 2) * 0.040

    suit_bonus = 0.035 if suited else 0.0
    gap = high - low
    connect_bonus = max(0, 0.025 - gap * 0.004)

    score = (high + low - 4) / 24.0 + suit_bonus + connect_bonus

    if high == 14:
        score += 0.09
    if high >= 13 and low >= 11:
        score += 0.06
    if high >= 13 and low >= 10:
        score += 0.02

    return min(max(score, 0.0), 1.0)


def monte_carlo_equity(
    my_hand: list[str],
    board: list[str],
    opp_revealed: list[str],
    n_sims: int,
) -> float:
    try:
        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board] if board else []
        opp_known = [eval7.Card(c) for c in opp_revealed] if opp_revealed else []

        known = set(my_cards + board_cards + opp_known)
        remaining = [c for c in eval7.Deck().cards if c not in known]

        wins = ties = total = 0
        board_need = 5 - len(board_cards)
        opp_need = 2 - len(opp_known)

        for _ in range(n_sims):
            random.shuffle(remaining)
            idx = 0
            opp_hand = list(opp_known)
            for _ in range(opp_need):
                opp_hand.append(remaining[idx])
                idx += 1
            sim_board = list(board_cards)
            for _ in range(board_need):
                sim_board.append(remaining[idx])
                idx += 1

            my_score = eval7.evaluate(sim_board + my_cards)
            opp_score = eval7.evaluate(sim_board + opp_hand)

            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            total += 1

        return (wins + ties * 0.5) / total if total else 0.5

    except Exception:
        return fast_hand_rank(my_hand)


class BoardTexture:
    def __init__(self, board: list[str]):
        self.board = board
        self.n = len(board)
        self.ranks = [RANK_VALUES.get(c[0], 0) for c in board]
        self.suits = [c[1] for c in board]

    @property
    def is_wet(self) -> bool:
        if self.n < 3:
            return False
        suit_counts = {s: self.suits.count(s) for s in set(self.suits)}
        max_suit = max(suit_counts.values()) if suit_counts else 0
        if max_suit >= 3:
            return True
        unique_ranks = sorted(set(self.ranks))
        if len(unique_ranks) >= 3:
            gaps = [unique_ranks[i + 1] - unique_ranks[i] for i in range(len(unique_ranks) - 1)]
            if sum(1 for g in gaps if g <= 2) >= 2:
                return True
        return False

    @property
    def is_dry(self) -> bool:
        return self.n >= 3 and not self.is_wet


class OpponentModel:
    def __init__(self):
        self.raise_count = 0
        self.call_count = 0
        self.check_count = 0
        self.fold_count = 0
        self.total_actions = 0
        self.bet_sizes: list[float] = []
        self.bids: list[int] = []

        self.prev_opp_wager = 0
        self.prev_my_wager = 0
        self.prev_street = ""
        self.opp_checked_this_street = False

    def update(self, state: PokerState) -> None:
        street = state.street
        if street == "auction":
            return

        opp_wager = state.opp_wager
        if street != self.prev_street:
            self.prev_opp_wager = 0
            self.prev_my_wager = 0
            self.prev_street = street
            self.opp_checked_this_street = False

        if opp_wager > self.prev_opp_wager:
            increase = opp_wager - self.prev_opp_wager
            pot_before = max(0, state.pot - increase)
            if self.prev_opp_wager == 0 and state.my_wager == 0:
                self.raise_count += 1
                if pot_before > 0:
                    self.bet_sizes.append(increase / pot_before)
            elif increase > state.my_wager - self.prev_opp_wager:
                self.raise_count += 1
                if pot_before > 0:
                    self.bet_sizes.append(increase / pot_before)
            else:
                self.call_count += 1
            self.total_actions += 1
        else:
            if state.cost_to_call == 0 and not self.opp_checked_this_street:
                self.check_count += 1
                self.total_actions += 1
                self.opp_checked_this_street = True

        self.prev_opp_wager = opp_wager
        self.prev_my_wager = state.my_wager

    def record_fold(self) -> None:
        self.fold_count += 1
        self.total_actions += 1

    def record_bid(self, bid: int) -> None:
        self.bids.append(bid)

    @property
    def fold_rate(self) -> float:
        if self.total_actions < 12:
            return 0.30
        return self.fold_count / self.total_actions

    @property
    def aggression(self) -> float:
        passive = self.call_count + self.check_count
        if passive == 0:
            return 2.0 if self.raise_count > 0 else 1.0
        return self.raise_count / passive

    @property
    def avg_bid(self) -> float:
        if not self.bids:
            return 10.0
        return sum(self.bids) / len(self.bids)

    @property
    def avg_bet_size(self) -> float:
        if not self.bet_sizes:
            return 0.55
        return sum(self.bet_sizes) / len(self.bet_sizes)


class Player(BaseBot):
    def __init__(self) -> None:
        self.opp = OpponentModel()

        self.hands_played = 0
        self.hand_aggression = 0
        self.preflop_equity = 0.5
        self.hand_equity = 0.5

        self._preflop_cache: dict[tuple, float] = {}
        self._equity_street = ""

        self.auction_pending = False
        self.auction_start_chips = 0
        self.last_bid = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preflop_eq(self, hand: list[str], tb: float) -> float:
        r1, r2 = hand[0][0], hand[1][0]
        suited = hand[0][1] == hand[1][1]
        ranks = tuple(sorted([r1, r2], key=lambda x: RANK_VALUES[x], reverse=True))
        key = (ranks[0], ranks[1], suited)
        if key not in self._preflop_cache:
            n = 220 if tb > 12.0 else 120
            self._preflop_cache[key] = monte_carlo_equity(hand, [], [], n)
        return self._preflop_cache[key]

    def _postflop_eq(self, state: PokerState, tb: float) -> float:
        if tb > 10.0:
            n = 160
        elif tb > 6.0:
            n = 90
        elif tb > 3.0:
            n = 50
        else:
            return fast_hand_rank(state.my_hand)
        return monte_carlo_equity(state.my_hand, state.board, state.opp_revealed_cards, n)

    def _pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0:
            return 0.0
        return cost / (state.pot + cost)

    def _bet_size(self, state: PokerState, equity: float, street: str) -> int:
        pot = state.pot
        opp_fold = self.opp.fold_rate

        if equity > 0.82:
            bet_frac = 0.85
        elif equity > 0.70:
            bet_frac = 0.65
        elif equity > 0.58:
            bet_frac = 0.52
        elif equity > 0.48:
            bet_frac = 0.40
        else:
            bet_frac = 0.30 if opp_fold > 0.35 else 0.40

        if street == "river":
            if equity > 0.80:
                bet_frac = max(bet_frac, 0.80)
            elif equity < 0.30 and opp_fold > 0.32:
                bet_frac = max(bet_frac, 0.55)

        target = int(pot * bet_frac) + state.my_wager
        min_raise, max_raise = state.raise_bounds
        return max(min_raise, min(target, max_raise))

    def _stack_bias(self, state: PokerState) -> float:
        ratio = (state.my_chips + 1) / (state.opp_chips + 1)
        if ratio > 1.2:
            return 0.02
        if ratio < 0.8:
            return -0.02
        return 0.0

    def _should_bluff(self, state: PokerState, street: str, equity: float) -> bool:
        fold_rate = self.opp.fold_rate
        base = 0.10 + (fold_rate - 0.30) * 0.6
        base = max(0.04, min(base, 0.45))

        if street in ("turn", "river"):
            base += 0.04
        if self.hand_aggression >= 2:
            base *= 0.35
        if state.pot > 700 and equity < 0.20:
            base *= 0.5
        if self.opp.aggression < 0.9:
            base *= 0.6
        return random.random() < base

    # ------------------------------------------------------------------
    # Auction
    # ------------------------------------------------------------------

    def _auction_bid(self, state: PokerState) -> ActionBid:
        equity = self.hand_equity
        pot = state.pot
        my_chips = state.my_chips

        if equity > 0.80:
            info_value = 0.10
        elif equity > 0.65:
            info_value = 0.06 + (equity - 0.65) * 0.5
        elif equity > 0.50:
            info_value = 0.09 + (0.65 - abs(equity - 0.575)) * 0.35
        elif equity > 0.38:
            info_value = 0.04
        else:
            info_value = 0.01

        base_bid = int(info_value * pot)

        if len(self.opp.bids) >= 6:
            opp_avg = self.opp.avg_bid
            if equity > 0.55:
                base_bid = max(base_bid, int(opp_avg) + 3)
            elif equity < 0.35:
                base_bid = 0
            if opp_avg < 4 and equity > 0.50:
                base_bid = max(base_bid, 4)
            if opp_avg > 12 and equity < 0.50:
                base_bid = 0
            if 8 <= opp_avg <= 12 and equity > 0.58:
                base_bid = max(base_bid, int(opp_avg) + 4)
            if opp_avg < 8 and equity > 0.75:
                base_bid = max(base_bid, 6)

        if equity > 0.70:
            cap = int(my_chips * 0.22)
        elif equity > 0.50:
            cap = int(my_chips * 0.12)
        else:
            cap = int(my_chips * 0.05)

        bid = max(0, min(base_bid, cap, my_chips))
        self.auction_pending = True
        self.auction_start_chips = my_chips
        self.last_bid = bid
        return ActionBid(bid)

    def _resolve_auction(self, state: PokerState) -> None:
        if not self.auction_pending:
            return
        self.auction_pending = False
        paid = max(0, self.auction_start_chips - state.my_chips)

        if paid > 0 and state.opp_revealed_cards:
            # We won (or tied) the auction. We can infer a bid.
            if paid == self.last_bid:
                self.opp.record_bid(self.last_bid)  # tie
            else:
                self.opp.record_bid(paid)  # winner pays opponent bid

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def _handle_preflop(self, state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise:
        equity = self.preflop_equity
        cost = state.cost_to_call
        in_pos = state.is_bb
        opp_agg = self.opp.aggression
        stack_bias = self._stack_bias(state)

        if cost > 0:
            pot_odds = self._pot_odds(state)

            if equity > 0.72 and state.can_act(ActionRaise):
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, "pre-flop"))

            if equity > 0.60 and state.can_act(ActionRaise) and random.random() < 0.30:
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, "pre-flop"))

            if equity > pot_odds + 0.05 + max(stack_bias, 0.0):
                return ActionCall()

            if in_pos and equity > pot_odds - 0.02 + max(stack_bias, 0.0):
                return ActionCall()

            if opp_agg > 2.2 and equity > 0.36 + max(stack_bias, 0.0):
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        if state.can_act(ActionRaise):
            if equity > 0.55:
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, "pre-flop"))
            if in_pos and equity > 0.38 and random.random() < 0.55:
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)
            if equity > 0.42 and random.random() < 0.40:
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)
            if equity < 0.30 and self._should_bluff(state, "pre-flop", equity):
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

        return ActionCheck()

    def _handle_postflop(self, game_info: GameInfo, state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise:
        street = state.street

        if self._equity_street != street:
            self.hand_equity = self._postflop_eq(state, game_info.time_bank)
            self._equity_street = street

        equity = self.hand_equity
        cost = state.cost_to_call
        pot_odds = self._pot_odds(state)
        opp_agg = self.opp.aggression
        texture = BoardTexture(state.board)
        stack_bias = self._stack_bias(state)

        if cost > 0:
            effective_odds = pot_odds * (0.82 if street in ("flop", "turn") else 1.0)
            big_bet = cost / max(state.pot, 1)

            if equity > 0.78 and state.can_act(ActionRaise):
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if equity > 0.63 and state.can_act(ActionRaise) and random.random() < 0.30:
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if equity > effective_odds + 0.03 + max(stack_bias, 0.0):
                return ActionCall()

            if street in ("flop", "turn") and equity > effective_odds - 0.03 + max(stack_bias, 0.0):
                if random.random() < 0.55:
                    return ActionCall()

            if opp_agg > 2.4 and equity > effective_odds - 0.05 + max(stack_bias, 0.0):
                return ActionCall()

            if (
                state.can_act(ActionRaise)
                and big_bet < 0.45
                and equity > 0.52
                and self.opp.avg_bet_size < 0.55
                and random.random() < 0.25
            ):
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if big_bet > 0.75 and equity < 0.60 + stack_bias:
                if state.can_act(ActionFold):
                    return ActionFold()
                return ActionCheck()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        if state.can_act(ActionRaise):
            if equity > 0.62:
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if equity > 0.50 and random.random() < 0.40:
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if equity < 0.32 and self._should_bluff(state, street, equity):
                self.hand_aggression += 1
                return ActionRaise(self._bet_size(state, equity, street))

            if texture.is_dry and self.opp.fold_rate > 0.32:
                if equity < 0.40 and random.random() < 0.22:
                    self.hand_aggression += 1
                    min_raise, _ = state.raise_bounds
                    return ActionRaise(min_raise)

            if texture.is_dry and equity > 0.42 and random.random() < 0.25:
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

        return ActionCheck()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.hand_aggression = 0
        self._equity_street = ""
        self.auction_pending = False
        self.auction_start_chips = 0
        self.last_bid = 0

        if game_info.time_bank > 4.0:
            self.preflop_equity = self._preflop_eq(current_state.my_hand, game_info.time_bank)
        else:
            self.preflop_equity = fast_hand_rank(current_state.my_hand)
        self.hand_equity = self.preflop_equity

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if current_state.payoff > 0 and not current_state.opp_revealed_cards:
            if self.hand_aggression > 0:
                self.opp.record_fold()

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        street = current_state.street

        if street != "auction":
            self.opp.update(current_state)
            self._resolve_auction(current_state)

        if street == "auction":
            return self._auction_bid(current_state)

        if street == "pre-flop":
            return self._handle_preflop(current_state)

        return self._handle_postflop(game_info, current_state)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
