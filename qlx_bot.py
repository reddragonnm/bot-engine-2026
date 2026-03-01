"""
QLX Bot - Enhanced Q-Learning Hybrid for IIT Pokerbots 2026.

Core ideas:
- Strong heuristic policy baseline (Apex/Ultra style logic)
- Q-table learns adjustments on top of heuristic scores
- Expanded state abstraction with board texture and stack pressure
- Softmax action selection with light optimism for unexplored actions
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random
import math


# =============================================================================
# Constants
# =============================================================================
NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10

RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

# Abstract actions
A_FOLD = 0
A_CHECK = 1
A_CALL = 2
A_RAISE_S = 3
A_RAISE_M = 4
A_RAISE_L = 5
A_BID_0 = 6
A_BID_L = 7
A_BID_M = 8
A_BID_H = 9

AUCTION_ACTIONS = [A_BID_0, A_BID_L, A_BID_M, A_BID_H]

# Q-learning
LEARNING_RATE = 0.12
DISCOUNT = 0.90
LAMBDA = 0.80
TEMP_START = 0.18
TEMP_END = 0.03
TEMP_DECAY = 700
EXPLORE_BONUS = 0.08

STREET_MAP = {"pre-flop": 0, "flop": 1, "auction": 2, "turn": 3, "river": 4}


# =============================================================================
# Equity
# =============================================================================

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
    my_hand: list[str], board: list[str], opp_revealed: list[str], n_sims: int = 150
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
            oh = list(opp_known)
            for _ in range(opp_need):
                oh.append(remaining[idx]); idx += 1
            sb = list(board_cards)
            for _ in range(board_need):
                sb.append(remaining[idx]); idx += 1
            ms = eval7.evaluate(sb + my_cards)
            os_ = eval7.evaluate(sb + oh)
            if ms > os_:
                wins += 1
            elif ms == os_:
                ties += 1
            total += 1
        return (wins + ties * 0.5) / total if total else 0.5
    except Exception:
        return fast_hand_rank(my_hand)


# =============================================================================
# Board texture
# =============================================================================

class BoardTexture:
    def __init__(self, board: list[str]):
        self.board = board
        self.n = len(board)
        self.ranks = [RANK_VALUES.get(c[0], 0) for c in board]
        self.suits = [c[1] for c in board]

    @property
    def is_paired(self) -> bool:
        return self.n >= 2 and len(set(self.ranks)) < self.n

    @property
    def is_wet(self) -> bool:
        if self.n < 3:
            return False
        suit_counts = {s: self.suits.count(s) for s in set(self.suits)}
        if suit_counts and max(suit_counts.values()) >= 3:
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


def texture_bin(board: list[str]) -> int:
    if not board or len(board) < 3:
        return 1
    tex = BoardTexture(board)
    if tex.is_wet:
        return 2
    if tex.is_dry:
        return 0
    return 1


# =============================================================================
# State discretization
# =============================================================================

def bin_equity(eq: float) -> int:
    if eq < 0.20: return 0
    if eq < 0.35: return 1
    if eq < 0.45: return 2
    if eq < 0.55: return 3
    if eq < 0.65: return 4
    if eq < 0.78: return 5
    return 6

def bin_pot(pot: int) -> int:
    r = pot / BIG_BLIND
    if r < 3: return 0
    if r < 10: return 1
    if r < 28: return 2
    return 3

def bin_cost(cost: int, pot: int) -> int:
    if cost <= 0: return 0
    if pot <= 0: return 2
    return 1 if cost / pot < 0.40 else 2

def bin_stack(my_chips: int, opp_chips: int) -> int:
    ratio = (my_chips + 1) / (opp_chips + 1)
    if ratio < 0.75: return 0
    if ratio > 1.30: return 2
    return 1


# =============================================================================
# Opponent model
# =============================================================================

class OpponentModel:
    def __init__(self) -> None:
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
    def avg_bet_size(self) -> float:
        if not self.bet_sizes:
            return 0.55
        return sum(self.bet_sizes) / len(self.bet_sizes)

    @property
    def avg_bid(self) -> float:
        if not self.bids:
            return 10.0
        return sum(self.bids) / len(self.bids)

    @property
    def bid_std(self) -> float:
        if len(self.bids) < 3:
            return 8.0
        avg = self.avg_bid
        var = sum((b - avg) ** 2 for b in self.bids) / len(self.bids)
        return math.sqrt(var)

    def type_bucket(self) -> int:
        fold = self.fold_rate
        agg = self.aggression
        tight = fold > 0.35
        loose = fold < 0.22
        aggressive = agg > 1.5
        passive = agg < 0.8
        if tight and aggressive:
            return 0
        if tight and passive:
            return 1
        if loose and aggressive:
            return 2
        if loose and passive:
            return 3
        return 1 if tight else 2


# =============================================================================
# Q-table
# =============================================================================

class QTable:
    def __init__(self):
        self.q: dict[tuple, dict[int, float]] = {}
        self.n: dict[tuple, dict[int, int]] = {}

    def get_adj(self, state: tuple, action: int) -> float:
        if state in self.q and action in self.q[state]:
            return self.q[state][action]
        return 0.0

    def get_count(self, state: tuple, action: int) -> int:
        if state in self.n and action in self.n[state]:
            return self.n[state][action]
        return 0

    def update(self, state: tuple, action: int, target: float, lr: float):
        if state not in self.q:
            self.q[state] = {}
            self.n[state] = {}
        if action not in self.q[state]:
            self.q[state][action] = 0.0
            self.n[state][action] = 0

        self.n[state][action] += 1
        eff_lr = lr / (1.0 + 0.004 * self.n[state][action])
        old = self.q[state][action]
        self.q[state][action] = old + eff_lr * (target - old)


# =============================================================================
# Bot
# =============================================================================

class Player(BaseBot):
    def __init__(self) -> None:
        self.q_table = QTable()
        self.opp = OpponentModel()
        self.temperature = TEMP_START

        # Per-hand
        self.hands_played = 0
        self.hand_equity = 0.5
        self.preflop_equity = 0.5
        self.hand_aggression = 0
        self.prev_street = ""

        # Trajectory: (q_state, action_id, equity, was_aggressive)
        self.trajectory: list[tuple[tuple, int, float, int]] = []

        self._preflop_cache: dict[tuple, float] = {}
        self.auction_pending = False
        self.auction_start_chips = 0
        self.last_bid = 0

    # ----- Equity -----

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

    # ----- State -----

    def _make_state(self, state: PokerState, equity: float, initiative: int) -> tuple:
        return (
            STREET_MAP.get(state.street, 0),
            bin_equity(equity),
            bin_pot(state.pot),
            bin_cost(state.cost_to_call, state.pot),
            1 if state.is_bb else 0,
            self.opp.type_bucket(),
            texture_bin(state.board),
            bin_stack(state.my_chips, state.opp_chips),
            initiative,
        )

    # ----- Legal Actions -----

    def _legal_actions(self, state: PokerState) -> list[int]:
        if state.street == "auction":
            return list(AUCTION_ACTIONS)
        legal = []
        if state.can_act(ActionFold):
            legal.append(A_FOLD)
        if state.can_act(ActionCheck):
            legal.append(A_CHECK)
        if state.can_act(ActionCall):
            legal.append(A_CALL)
        if state.can_act(ActionRaise):
            legal.extend([A_RAISE_S, A_RAISE_M, A_RAISE_L])
        return legal if legal else [A_CHECK]

    # =========================================================================
    # Heuristic policy
    # =========================================================================

    def _heuristic_scores(
        self, state: PokerState, equity: float, legal: list[int], initiative: int
    ) -> dict[int, float]:
        scores: dict[int, float] = {a: -3.0 for a in legal}

        street = state.street
        cost = state.cost_to_call
        pot = state.pot
        pot_odds = cost / (pot + cost) if cost > 0 and pot + cost > 0 else 0.0
        opp_fold = self.opp.fold_rate
        opp_agg = self.opp.aggression
        opp_loose = self.opp.type_bucket() in (2, 3)
        texture = texture_bin(state.board)
        in_pos = state.is_bb

        # ---- AUCTION ----
        if street == "auction":
            if A_BID_0 in legal:
                scores[A_BID_0] = 0.15 if equity < 0.38 else -0.05
            if A_BID_L in legal:
                scores[A_BID_L] = (equity - 0.40) * 0.55
            if A_BID_M in legal:
                scores[A_BID_M] = (equity - 0.52) * 0.75
            if A_BID_H in legal:
                scores[A_BID_H] = (equity - 0.68) * 0.95
            return scores

        # ---- PREFLOP ----
        if street == "pre-flop":
            if cost > 0:
                if A_FOLD in legal:
                    scores[A_FOLD] = -0.05
                if A_CALL in legal:
                    ev = equity - pot_odds
                    if in_pos:
                        ev += 0.04
                    if opp_agg > 2.0:
                        ev += 0.05
                    if opp_loose:
                        ev -= 0.02
                    scores[A_CALL] = ev * 0.9
                if A_RAISE_S in legal:
                    scores[A_RAISE_S] = (equity - 0.56) * 1.2 + opp_fold * 0.15
                if A_RAISE_M in legal:
                    scores[A_RAISE_M] = (equity - 0.63) * 1.4 + opp_fold * 0.20
                if A_RAISE_L in legal:
                    scores[A_RAISE_L] = (equity - 0.71) * 1.6 + opp_fold * 0.25
            else:
                if A_CHECK in legal:
                    scores[A_CHECK] = 0.02
                if A_RAISE_S in legal:
                    s = (equity - 0.40) * 0.9 + opp_fold * 0.14
                    if equity < 0.30:
                        s = max(s, opp_fold * 0.35 - 0.10)
                    scores[A_RAISE_S] = s
                if A_RAISE_M in legal:
                    scores[A_RAISE_M] = (equity - 0.50) * 1.1 + opp_fold * 0.18
                if A_RAISE_L in legal:
                    scores[A_RAISE_L] = (equity - 0.60) * 1.3 + opp_fold * 0.22
            return scores

        # ---- POSTFLOP ----
        implied_mult = 0.82 if street in ("flop", "turn") else 1.0
        is_draw_street = street in ("flop", "turn")

        if cost > 0:
            eff_odds = pot_odds * implied_mult
            if A_FOLD in legal:
                scores[A_FOLD] = -0.05
            if A_CALL in legal:
                ev = equity - eff_odds
                if opp_agg > 2.4:
                    ev += 0.05
                if is_draw_street and ev > -0.06:
                    ev += 0.03
                scores[A_CALL] = ev * 0.95
            if A_RAISE_S in legal:
                s = (equity - 0.52) * 1.1 + opp_fold * 0.16
                if is_draw_street and equity > 0.30 and equity < 0.52:
                    s = max(s, opp_fold * 0.24 - 0.08)
                scores[A_RAISE_S] = s
            if A_RAISE_M in legal:
                scores[A_RAISE_M] = (equity - 0.60) * 1.3 + opp_fold * 0.22
            if A_RAISE_L in legal:
                scores[A_RAISE_L] = (equity - 0.72) * 1.5 + opp_fold * 0.28
        else:
            if A_CHECK in legal:
                scores[A_CHECK] = 0.03
            if A_RAISE_S in legal:
                if equity > 0.50:
                    s = (equity - 0.42) * 0.95
                elif equity < 0.32:
                    bluff_ev = opp_fold * (0.32 if texture == 2 else 0.26)
                    aggression_penalty = 0.12 if self.hand_aggression >= 2 else 0.0
                    s = bluff_ev - 0.12 - aggression_penalty
                else:
                    s = -0.04
                scores[A_RAISE_S] = s
            if A_RAISE_M in legal:
                scores[A_RAISE_M] = (equity - 0.56) * 1.1 + opp_fold * 0.18
            if A_RAISE_L in legal:
                if equity > 0.80:
                    s = (equity - 0.52) * 1.6
                elif street == "river" and equity < 0.30:
                    s = opp_fold * 0.42 - 0.20
                else:
                    s = (equity - 0.66) * 1.4 + opp_fold * 0.22
                scores[A_RAISE_L] = s

            if initiative and street == "flop" and A_RAISE_S in legal:
                scores[A_RAISE_S] = max(scores[A_RAISE_S], 0.04)

        return scores

    # =========================================================================
    # Action selection
    # =========================================================================

    def _select_action(
        self, q_state: tuple, legal: list[int], heuristic_scores: dict[int, float]
    ) -> int:
        combined = {}
        for a in legal:
            base = heuristic_scores.get(a, -3.0)
            adj = self.q_table.get_adj(q_state, a)
            count = self.q_table.get_count(q_state, a)
            explore = EXPLORE_BONUS / math.sqrt(1 + count)
            combined[a] = base + adj + explore

        temp = max(self.temperature, 0.005)
        max_v = max(combined.values()) if combined else 0.0

        weights = {}
        for a, v in combined.items():
            weights[a] = math.exp((v - max_v) / temp)

        total = sum(weights.values())
        if total <= 0:
            return max(combined, key=lambda a: combined[a])

        r = random.random() * total
        cum = 0.0
        for a, w in weights.items():
            cum += w
            if r <= cum:
                return a
        return list(weights.keys())[-1]

    # =========================================================================
    # Convert to concrete action
    # =========================================================================

    def _auction_bid_amount(self, action: int, state: PokerState, equity: float) -> int:
        if action == A_BID_0:
            return 0

        pot = state.pot
        my_chips = state.my_chips
        opp_avg = self.opp.avg_bid
        opp_std = self.opp.bid_std

        if equity > 0.80:
            info_value = 0.10
        elif equity > 0.65:
            info_value = 0.06 + (equity - 0.65) * 0.5
        elif equity > 0.50:
            info_value = 0.08 + (0.65 - abs(equity - 0.575)) * 0.3
        elif equity > 0.38:
            info_value = 0.04
        else:
            info_value = 0.01

        base_bid = int(info_value * pot)

        if action == A_BID_L:
            base_bid = max(base_bid, int(opp_avg * 0.6))
        elif action == A_BID_M:
            base_bid = max(base_bid, int(opp_avg + opp_std * 0.2) + 1)
        elif action == A_BID_H:
            base_bid = max(base_bid, int(opp_avg + opp_std * 0.6) + 2)

        if equity > 0.70:
            cap = int(my_chips * 0.20)
        elif equity > 0.50:
            cap = int(my_chips * 0.12)
        else:
            cap = int(my_chips * 0.05)

        bid = max(0, min(base_bid, cap, my_chips))
        return bid

    def _raise_amount(self, action: int, state: PokerState, equity: float) -> int:
        min_r, max_r = state.raise_bounds
        pot = state.pot
        opp_fold = self.opp.fold_rate
        opp_loose = self.opp.type_bucket() in (2, 3)

        if action == A_RAISE_S:
            if equity > 0.60:
                frac = 0.45
            else:
                frac = 0.33 if opp_loose else 0.38
        elif action == A_RAISE_M:
            frac = 0.65
        else:
            if equity > 0.82:
                frac = 0.90
            elif state.street == "river" and equity < 0.32 and opp_fold > 0.32:
                frac = 0.55
            else:
                frac = 0.85

        target = int(pot * frac) + state.my_wager
        return max(min_r, min(target, max_r))

    def _to_concrete(
        self, action: int, state: PokerState, equity: float
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        if action in AUCTION_ACTIONS:
            return ActionBid(self._auction_bid_amount(action, state, equity))

        if action == A_FOLD:
            return ActionFold() if state.can_act(ActionFold) else ActionCheck()
        if action == A_CHECK:
            if state.can_act(ActionCheck):
                return ActionCheck()
            if state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()
        if action == A_CALL:
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        if state.can_act(ActionRaise):
            amount = self._raise_amount(action, state, equity)
            self.hand_aggression += 1
            return ActionRaise(amount)

        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.hand_aggression = 0
        self.prev_street = ""
        self.trajectory = []
        self.auction_pending = False
        self.auction_start_chips = 0
        self.last_bid = 0

        t = min(self.hands_played, TEMP_DECAY)
        self.temperature = TEMP_START + (TEMP_END - TEMP_START) * (t / TEMP_DECAY)

        if game_info.time_bank > 5.0:
            self.preflop_equity = self._preflop_eq(current_state.my_hand, game_info.time_bank)
        else:
            self.preflop_equity = fast_hand_rank(current_state.my_hand)
        self.hand_equity = self.preflop_equity

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff
        reward = payoff / STARTING_STACK

        if payoff > 0 and not current_state.opp_revealed_cards:
            if self.hand_aggression > 0:
                self.opp.record_fold()

        # Small bluff bonus for winning with low equity
        if payoff > 0 and self.trajectory:
            avg_eq = sum(t[2] for t in self.trajectory) / len(self.trajectory)
            if avg_eq < 0.42 and self.hand_aggression > 0:
                reward += 0.015

        n = len(self.trajectory)
        for i, (s, a, _eq, _aggr) in enumerate(self.trajectory):
            weight = (DISCOUNT * LAMBDA) ** (n - 1 - i)
            self.q_table.update(s, a, reward * weight, LEARNING_RATE)

    # =========================================================================
    # Main
    # =========================================================================

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        street = current_state.street

        if street != "auction":
            self.opp.update(current_state)

        # Equity
        if street == "pre-flop":
            equity = self.preflop_equity
        elif street == "auction":
            equity = self.hand_equity
        else:
            equity = self._postflop_eq(current_state, game_info.time_bank)
            self.hand_equity = equity

        initiative = 1 if self.hand_aggression > 0 else 0
        q_state = self._make_state(current_state, equity, initiative)
        legal = self._legal_actions(current_state)
        h_scores = self._heuristic_scores(current_state, equity, legal, initiative)

        action = self._select_action(q_state, legal, h_scores)
        self.trajectory.append((q_state, action, equity, 1 if self.hand_aggression > 0 else 0))

        return self._to_concrete(action, current_state, equity)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
