"""
Self-learning Q-table poker bot for IIT Pokerbots 2026.

Architecture:
  The bot uses a STRONG heuristic baseline (equivalent to main.py) for all
  decisions, then learns Q-table corrections on top to exploit opponent
  tendencies over the 1000-round match.

  Approach:
  1. Compute equity via Monte Carlo (with caching and time awareness)
  2. The heuristic produces an action AND a confidence score
  3. Q-table maps (discretized state, abstract action) -> adjustment value
  4. Softmax selection over: heuristic_score[a] + q_adjustment[a]
  5. Monte Carlo return updates the Q-adjustments at hand end
  6. Low temperature means Q only overrides heuristic when it's confident

  This ensures:
  - Round 1 plays as strong as the heuristic bot
  - By round 200-300, Q-learning starts exploiting opponent patterns
  - By round 500+, the bot has learned significant exploits

  Key learning signals:
  - Opponent folds too much -> increase bluff frequency
  - Opponent calls too much -> tighten value range, reduce bluffs
  - Opponent overbids auction -> let them waste chips
  - Opponent underbids auction -> win info cheaply
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
LEARNING_RATE = 0.10
DISCOUNT = 0.90
TEMP_START = 0.15   # low temperature: trust heuristic early
TEMP_END = 0.02     # very greedy late
TEMP_DECAY = 600

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
        return 0.5 + (high - 2) * 0.038
    suit_bonus = 0.03 if suited else 0.0
    gap = high - low
    connect_bonus = max(0, 0.02 - gap * 0.003)
    score = (high + low - 4) / 24.0 + suit_bonus + connect_bonus
    if high == 14:
        score += 0.08
    if high >= 13 and low >= 11:
        score += 0.05
    return min(max(score, 0.0), 1.0)


def monte_carlo_equity(
    my_hand: list[str], board: list[str],
    opp_revealed: list[str], n_sims: int = 150,
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
            if ms > os_: wins += 1
            elif ms == os_: ties += 1
            total += 1
        return (wins + ties * 0.5) / total if total else 0.5
    except Exception:
        return fast_hand_rank(my_hand)


# =============================================================================
# State discretization (compact: ~1440 states)
# =============================================================================

def bin_equity(eq: float) -> int:
    """6 bins."""
    if eq < 0.25: return 0
    if eq < 0.40: return 1
    if eq < 0.50: return 2
    if eq < 0.60: return 3
    if eq < 0.75: return 4
    return 5

def bin_pot(pot: int) -> int:
    """4 bins."""
    r = pot / BIG_BLIND
    if r < 3:  return 0
    if r < 10: return 1
    if r < 30: return 2
    return 3

def bin_cost(cost: int, pot: int) -> int:
    """3 bins."""
    if cost <= 0: return 0
    if pot <= 0:  return 2
    return 1 if cost / pot < 0.40 else 2

def bin_opp(agg: float, fold_rate: float) -> int:
    """4 types."""
    a = 1 if agg >= 1.2 else 0
    f = 1 if fold_rate >= 0.30 else 0
    return a * 2 + f


# =============================================================================
# Q-Table (stores adjustments, not absolute values)
# =============================================================================

class QTable:
    def __init__(self):
        self.q: dict[tuple, dict[int, float]] = {}
        self.n: dict[tuple, dict[int, int]] = {}

    def get_adj(self, state: tuple, action: int) -> float:
        """Get Q-adjustment for (state, action). Default 0 = no adjustment."""
        if state in self.q and action in self.q[state]:
            return self.q[state][action]
        return 0.0

    def update(self, state: tuple, action: int, target: float, lr: float):
        if state not in self.q:
            self.q[state] = {}
            self.n[state] = {}
        if action not in self.q[state]:
            self.q[state][action] = 0.0
            self.n[state][action] = 0

        self.n[state][action] += 1
        eff_lr = lr / (1.0 + 0.003 * self.n[state][action])
        old = self.q[state][action]
        self.q[state][action] = old + eff_lr * (target - old)


# =============================================================================
# Bot
# =============================================================================

class Player(BaseBot):
    def __init__(self) -> None:
        self.q_table = QTable()
        self.temperature = TEMP_START

        # Opponent model
        self.opp_fold_count = 0
        self.opp_raise_count = 0
        self.opp_call_count = 0
        self.opp_check_count = 0
        self.opp_total_actions = 0
        self.opp_bids: list[int] = []

        # Per-hand
        self.hands_played = 0
        self.hand_equity = 0.5
        self.preflop_equity = 0.5
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""

        # Trajectory: (q_state, action_id)
        self.trajectory: list[tuple[tuple, int]] = []

        self._preflop_cache: dict[tuple, float] = {}

    # ----- Opponent Model -----

    def _opp_fold_rate(self) -> float:
        if self.opp_total_actions < 10: return 0.30
        return self.opp_fold_count / self.opp_total_actions

    def _opp_aggression(self) -> float:
        passive = self.opp_call_count + self.opp_check_count
        if passive == 0:
            return 2.0 if self.opp_raise_count > 0 else 1.0
        return self.opp_raise_count / passive

    def _opp_avg_bid(self) -> float:
        if not self.opp_bids: return 50.0
        return sum(self.opp_bids) / len(self.opp_bids)

    def _infer_opp(self, state: PokerState) -> None:
        street = state.street
        ow = state.opp_wager
        if street != self.prev_street:
            self.prev_opp_wager = 0
            self.prev_street = street
        if ow > self.prev_opp_wager:
            inc = ow - self.prev_opp_wager
            if self.prev_opp_wager == 0 and state.my_wager == 0:
                self.opp_raise_count += 1
            elif inc > state.my_wager - self.prev_opp_wager:
                self.opp_raise_count += 1
            else:
                self.opp_call_count += 1
            self.opp_total_actions += 1
        self.prev_opp_wager = ow

    # ----- Equity -----

    def _preflop_eq(self, hand: list[str], tb: float) -> float:
        r1, r2 = hand[0][0], hand[1][0]
        suited = hand[0][1] == hand[1][1]
        ranks = tuple(sorted([r1, r2], key=lambda x: RANK_VALUES[x], reverse=True))
        key = (ranks[0], ranks[1], suited)
        if key not in self._preflop_cache:
            n = 200 if tb > 12.0 else 100
            self._preflop_cache[key] = monte_carlo_equity(hand, [], [], n)
        return self._preflop_cache[key]

    def _postflop_eq(self, state: PokerState, tb: float) -> float:
        if tb > 10.0:   n = 150
        elif tb > 5.0:  n = 80
        elif tb > 2.0:  n = 40
        else:           return fast_hand_rank(state.my_hand)
        return monte_carlo_equity(
            state.my_hand, state.board, state.opp_revealed_cards, n
        )

    # ----- State -----

    def _make_state(self, state: PokerState, equity: float) -> tuple:
        """5 * 6 * 4 * 3 * 2 * 4 = 2880 states."""
        return (
            STREET_MAP.get(state.street, 0),
            bin_equity(equity),
            bin_pot(state.pot),
            bin_cost(state.cost_to_call, state.pot),
            1 if state.is_bb else 0,
            bin_opp(self._opp_aggression(), self._opp_fold_rate()),
        )

    # ----- Legal Actions -----

    def _legal_actions(self, state: PokerState) -> list[int]:
        if state.street == "auction":
            return list(AUCTION_ACTIONS)
        legal = []
        if state.can_act(ActionFold):  legal.append(A_FOLD)
        if state.can_act(ActionCheck): legal.append(A_CHECK)
        if state.can_act(ActionCall):  legal.append(A_CALL)
        if state.can_act(ActionRaise):
            legal.extend([A_RAISE_S, A_RAISE_M, A_RAISE_L])
        return legal if legal else [A_CHECK]

    # =========================================================================
    # HEURISTIC POLICY (strong baseline, mirrors main.py logic)
    # =========================================================================

    def _heuristic_scores(
        self, state: PokerState, equity: float, legal: list[int]
    ) -> dict[int, float]:
        """
        Return a score for each legal action based on the heuristic policy.
        Higher = better. These are the "base Q-values" that Q-learning adjusts.
        Scores are in a wider range to give Q-learning meaningful signal.
        """
        scores: dict[int, float] = {}
        for a in legal:
            scores[a] = -3.0  # default: strongly discouraged

        street = state.street
        cost = state.cost_to_call
        pot = state.pot
        pot_odds = cost / (pot + cost) if cost > 0 and pot + cost > 0 else 0.0
        opp_fold = self._opp_fold_rate()
        opp_agg = self._opp_aggression()
        in_pos = state.is_bb

        # ---- AUCTION ----
        if street == "auction":
            if A_BID_0 in legal:
                scores[A_BID_0] = 0.1 if equity < 0.45 else -0.1
            if A_BID_L in legal:
                scores[A_BID_L] = (equity - 0.40) * 0.4
            if A_BID_M in legal:
                scores[A_BID_M] = (equity - 0.50) * 0.6
            if A_BID_H in legal:
                scores[A_BID_H] = (equity - 0.65) * 0.8
            return scores

        # ---- PREFLOP ----
        if street == "pre-flop":
            if cost > 0:
                # Facing raise
                if A_FOLD in legal:
                    # Folding is baseline - slightly negative
                    scores[A_FOLD] = -0.05
                if A_CALL in legal:
                    ev = equity - pot_odds
                    if in_pos: ev += 0.04
                    if opp_agg > 2.0: ev += 0.06
                    scores[A_CALL] = ev * 0.8
                if A_RAISE_S in legal:
                    # Mirror main.py: re-raise with equity > 0.58
                    s = (equity - 0.55) * 1.2 + opp_fold * 0.15
                    scores[A_RAISE_S] = s
                if A_RAISE_M in legal:
                    s = (equity - 0.62) * 1.4 + opp_fold * 0.20
                    scores[A_RAISE_M] = s
                if A_RAISE_L in legal:
                    s = (equity - 0.70) * 1.6 + opp_fold * 0.25
                    scores[A_RAISE_L] = s
            else:
                # No cost - can open raise or check
                if A_CHECK in legal:
                    scores[A_CHECK] = 0.0
                if A_RAISE_S in legal:
                    s = (equity - 0.40) * 0.8 + opp_fold * 0.12
                    # Bluff value: small raise with weak hands when opp folds a lot
                    if equity < 0.30:
                        bluff_ev = opp_fold * 0.35
                        s = max(s, bluff_ev - 0.10)
                    scores[A_RAISE_S] = s
                if A_RAISE_M in legal:
                    scores[A_RAISE_M] = (equity - 0.50) * 1.0 + opp_fold * 0.18
                if A_RAISE_L in legal:
                    scores[A_RAISE_L] = (equity - 0.58) * 1.2 + opp_fold * 0.22
            return scores

        # ---- POSTFLOP (flop, turn, river) ----
        implied_mult = 0.82 if street in ("flop", "turn") else 1.0
        is_draw_street = street in ("flop", "turn")

        if cost > 0:
            eff_odds = pot_odds * implied_mult
            if A_FOLD in legal:
                scores[A_FOLD] = -0.05
            if A_CALL in legal:
                ev = equity - eff_odds
                if opp_agg > 2.5: ev += 0.05
                # Drawing hand bonus on flop/turn
                if is_draw_street and ev > -0.06:
                    ev += 0.03
                scores[A_CALL] = ev * 0.9
            if A_RAISE_S in legal:
                # Value raise or semi-bluff
                s = (equity - 0.52) * 1.0 + opp_fold * 0.15
                # Semi-bluff on flop/turn with decent equity
                if is_draw_street and equity > 0.32 and equity < 0.52:
                    bluff_ev = opp_fold * 0.25
                    s = max(s, bluff_ev - 0.08)
                scores[A_RAISE_S] = s
            if A_RAISE_M in legal:
                scores[A_RAISE_M] = (equity - 0.60) * 1.3 + opp_fold * 0.22
            if A_RAISE_L in legal:
                scores[A_RAISE_L] = (equity - 0.72) * 1.5 + opp_fold * 0.28
        else:
            if A_CHECK in legal:
                scores[A_CHECK] = 0.0
            if A_RAISE_S in legal:
                # Value bet or bluff
                if equity > 0.50:
                    # Thin value bet (mirrors main.py: bet at eq > 0.50)
                    s = (equity - 0.42) * 0.9
                elif equity < 0.32:
                    # Bluff (mirrors main.py _should_bluff logic)
                    bluff_ev = opp_fold * 0.30
                    aggression_penalty = 0.12 if self.hand_aggression >= 2 else 0.0
                    s = bluff_ev - 0.12 - aggression_penalty
                else:
                    # Marginal hand: lean towards check
                    s = -0.05
                scores[A_RAISE_S] = s
            if A_RAISE_M in legal:
                s = (equity - 0.55) * 1.1 + opp_fold * 0.18
                scores[A_RAISE_M] = s
            if A_RAISE_L in legal:
                # Polarized: strong value or big bluff
                if equity > 0.78:
                    s = (equity - 0.50) * 1.5
                elif street == "river" and equity < 0.30:
                    s = opp_fold * 0.40 - 0.20
                else:
                    s = (equity - 0.65) * 1.4 + opp_fold * 0.22
                scores[A_RAISE_L] = s

        return scores

    # =========================================================================
    # Action selection: softmax over (heuristic + Q-adjustment)
    # =========================================================================

    def _select_action(
        self, q_state: tuple, legal: list[int],
        heuristic_scores: dict[int, float],
    ) -> int:
        """Softmax over combined scores."""
        combined = {}
        for a in legal:
            base = heuristic_scores.get(a, -3.0)
            if base < -2.0:
                continue  # skip impossible/terrible actions
            adj = self.q_table.get_adj(q_state, a)
            combined[a] = base + adj

        if not combined:
            return random.choice(legal)

        temp = max(self.temperature, 0.005)
        max_v = max(combined.values())

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

    def _to_concrete(
        self, action: int, state: PokerState, equity: float
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        # Auction
        if action == A_BID_0:
            return ActionBid(0)
        if action in (A_BID_L, A_BID_M, A_BID_H):
            my_chips = state.my_chips
            pot = state.pot
            opp_avg = self._opp_avg_bid()

            if action == A_BID_L:
                bid = max(1, int(pot * 0.02))
            elif action == A_BID_M:
                bid = max(2, int(pot * 0.08))
                if len(self.opp_bids) >= 5:
                    bid = max(bid, int(opp_avg * 0.8))
            else:
                bid = max(5, int(pot * 0.15))
                if len(self.opp_bids) >= 5:
                    bid = max(bid, int(opp_avg * 1.15) + 1)

            cap = int(my_chips * 0.25) if equity > 0.7 else int(my_chips * 0.15)
            bid = max(0, min(bid, cap, my_chips))
            return ActionBid(bid)

        if action == A_FOLD:
            return ActionFold() if state.can_act(ActionFold) else ActionCheck()
        if action == A_CHECK:
            if state.can_act(ActionCheck): return ActionCheck()
            if state.can_act(ActionCall): return ActionCall()
            return ActionFold()
        if action == A_CALL:
            if state.can_act(ActionCall): return ActionCall()
            if state.can_act(ActionCheck): return ActionCheck()
            return ActionFold()

        # Raises
        if state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            pot = state.pot
            street = state.street

            if action == A_RAISE_S:
                if equity > 0.55:
                    target = int(pot * 0.45) + state.my_wager
                else:
                    # Bluff sizing: smaller
                    target = int(pot * 0.33) + state.my_wager
            elif action == A_RAISE_M:
                target = int(pot * 0.65) + state.my_wager
            else:
                if equity > 0.85:
                    target = int(pot * 0.85) + state.my_wager
                elif street == "river" and equity > 0.8:
                    target = int(pot * 1.0) + state.my_wager
                elif street == "river" and equity < 0.35:
                    # Bluff overbet on river
                    target = int(pot * 0.55) + state.my_wager
                else:
                    target = int(pot * 0.90) + state.my_wager

            amount = max(min_r, min(target, max_r))
            self.hand_aggression += 1
            return ActionRaise(amount)

        if state.can_act(ActionCall): return ActionCall()
        if state.can_act(ActionCheck): return ActionCheck()
        return ActionFold()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.trajectory = []

        t = min(self.hands_played, TEMP_DECAY)
        self.temperature = TEMP_START + (TEMP_END - TEMP_START) * (t / TEMP_DECAY)

        if game_info.time_bank > 5.0:
            self.preflop_equity = self._preflop_eq(
                current_state.my_hand, game_info.time_bank
            )
        else:
            self.preflop_equity = fast_hand_rank(current_state.my_hand)
        self.hand_equity = self.preflop_equity

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff
        reward = payoff / STARTING_STACK

        # MC update
        n = len(self.trajectory)
        for i, (s, a) in enumerate(self.trajectory):
            discount = DISCOUNT ** (n - 1 - i)
            self.q_table.update(s, a, reward * discount, LEARNING_RATE)

        # Track opponent folds
        if payoff > 0 and not current_state.opp_revealed_cards:
            if self.hand_aggression > 0:
                self.opp_fold_count += 1
                self.opp_total_actions += 1

    # =========================================================================
    # Main
    # =========================================================================

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        street = current_state.street

        if street != "auction":
            self._infer_opp(current_state)

        # Equity
        if street == "pre-flop":
            equity = self.preflop_equity
        elif street == "auction":
            equity = self.hand_equity
        else:
            equity = self._postflop_eq(current_state, game_info.time_bank)
            self.hand_equity = equity

        q_state = self._make_state(current_state, equity)
        legal = self._legal_actions(current_state)
        h_scores = self._heuristic_scores(current_state, equity, legal)

        action = self._select_action(q_state, legal, h_scores)
        self.trajectory.append((q_state, action))

        return self._to_concrete(action, current_state, equity)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
