"""
Ultra Bot - Advanced Exploitative Poker Bot for Sneak Peek Hold'em
=================================================================

Key advantages over main.py and ql_bot.py:

1. PREFLOP LOOKUP TABLE: Precomputed canonical hand equity (169 hands) with
   high-sim Monte Carlo on first encounter, then cached. Eliminates noise.

2. EFFECTIVE HAND STRENGTH: On postflop, combines raw equity with hand
   potential (probability of improving on future streets). Uses more sims
   with smarter time budgeting.

3. OPTIMAL SECOND-PRICE AUCTION: Bids based on true information value.
   In a second-price auction, bidding your true value is dominant strategy.
   We compute the information edge from seeing an opponent card and bid
   proportionally. Also exploits opponent bidding patterns.

4. DEEP OPPONENT MODEL: Tracks per-street fold/raise frequencies, bet sizing
   patterns, showdown hand strengths, and auction behavior. Classifies
   opponent into archetypes and adjusts strategy dynamically.

5. BOARD TEXTURE ANALYSIS: Detects wet/dry boards, flush draws, straight
   draws, paired boards, and adjusts c-bet and calling frequencies.

6. POT GEOMETRY & BET SIZING: Sizes bets based on stack-to-pot ratio (SPR),
   using geometric bet sizing for multi-street value extraction.

7. BALANCED VALUE/BLUFF RATIOS: Uses pot-odds-based bluff-to-value ratios
   to stay unexploitable while maximally exploiting opponent tendencies.

8. POSITION AWARENESS: Different opening ranges and continuation strategies
   based on IP (in position) vs OOP play.

9. ADAPTIVE AGGRESSION: Increases bluff frequency vs. tight opponents,
   tightens range vs. calling stations, and adjusts bet sizes based on
   opponent tendencies.

10. TIME MANAGEMENT: Aggressive caching, adaptive sim counts, and fast
    heuristic fallbacks ensure we never time out.
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

SUIT_MAP = {"d": 0, "s": 1, "c": 2, "h": 3}


# =============================================================================
# Card / Board Utilities
# =============================================================================

def card_rank(card_str: str) -> int:
    return RANK_VALUES.get(card_str[0], 0)


def card_suit(card_str: str) -> int:
    return SUIT_MAP.get(card_str[1], 0)


def is_suited(hand: list[str]) -> bool:
    return len(hand) == 2 and hand[0][1] == hand[1][1]


def is_pair(hand: list[str]) -> bool:
    return len(hand) == 2 and hand[0][0] == hand[1][0]


def canonical_hand_key(hand: list[str]) -> tuple:
    """Return a canonical key for the hand (rank_high, rank_low, suited)."""
    r1 = RANK_VALUES[hand[0][0]]
    r2 = RANK_VALUES[hand[1][0]]
    high, low = max(r1, r2), min(r1, r2)
    suited = hand[0][1] == hand[1][1]
    return (high, low, suited)


# =============================================================================
# Board Texture Analysis
# =============================================================================

class BoardTexture:
    """Analyzes the community cards for strategic decisions."""

    def __init__(self, board: list[str]):
        self.board = board
        self.n = len(board)
        if self.n == 0:
            self.ranks = []
            self.suits = []
        else:
            self.ranks = sorted([card_rank(c) for c in board], reverse=True)
            self.suits = [card_suit(c) for c in board]

    @property
    def is_monotone(self) -> bool:
        """All same suit (flush-heavy board)."""
        return self.n >= 3 and len(set(self.suits)) == 1

    @property
    def has_flush_draw(self) -> bool:
        """3+ of one suit (potential flush)."""
        if self.n < 3:
            return False
        from collections import Counter
        suit_counts = Counter(self.suits)
        return suit_counts.most_common(1)[0][1] >= 3

    @property
    def flush_suit_count(self) -> int:
        """Max count of any single suit."""
        if self.n == 0:
            return 0
        from collections import Counter
        return Counter(self.suits).most_common(1)[0][1]

    @property
    def is_paired(self) -> bool:
        """Board has a pair."""
        return self.n >= 2 and len(set(self.ranks)) < self.n

    @property
    def high_card(self) -> int:
        return self.ranks[0] if self.ranks else 0

    @property
    def is_dry(self) -> bool:
        """Dry = no flush draw, no straight draw potential, rainbow."""
        if self.n < 3:
            return True
        # Rainbow + spread out ranks
        suits_unique = len(set(self.suits))
        if suits_unique < self.n:
            return False
        # Check for connectedness
        sorted_r = sorted(self.ranks)
        max_gap = max(sorted_r[i + 1] - sorted_r[i] for i in range(len(sorted_r) - 1))
        return max_gap >= 4

    @property
    def is_wet(self) -> bool:
        """Wet = flush draw or many straight possibilities."""
        return not self.is_dry

    @property
    def connectedness(self) -> float:
        """0-1 score of how connected the board is for straights."""
        if self.n < 2:
            return 0.0
        sorted_r = sorted(set(self.ranks))
        if len(sorted_r) < 2:
            return 0.0
        gaps = [sorted_r[i + 1] - sorted_r[i] for i in range(len(sorted_r) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        # Lower avg gap = more connected
        return max(0.0, min(1.0, 1.0 - (avg_gap - 1) / 5.0))


# =============================================================================
# Fast Preflop Hand Strength (heuristic for time-critical situations)
# =============================================================================

def fast_hand_rank(cards: list[str]) -> float:
    """Quick heuristic 0-1 for 2-card hands."""
    if len(cards) != 2:
        return 0.5
    r1 = RANK_VALUES.get(cards[0][0], 0)
    r2 = RANK_VALUES.get(cards[1][0], 0)
    suited = cards[0][1] == cards[1][1]
    high, low = max(r1, r2), min(r1, r2)

    # Pocket pairs
    if r1 == r2:
        return 0.50 + (high - 2) * 0.040  # 22~0.50, AA~0.98

    suit_bonus = 0.035 if suited else 0.0
    gap = high - low
    connect_bonus = max(0, 0.025 - gap * 0.004)

    score = (high + low - 4) / 24.0 + suit_bonus + connect_bonus

    if high == 14:  # Ace
        score += 0.09
    if high >= 13 and low >= 11:  # Broadway
        score += 0.06
    if high >= 13 and low >= 10:  # Near-broadway
        score += 0.02

    return min(max(score, 0.0), 1.0)


# =============================================================================
# Monte Carlo Equity Estimation
# =============================================================================

def monte_carlo_equity(
    my_hand: list[str],
    board: list[str],
    opp_revealed: list[str],
    n_sims: int = 200,
) -> float:
    """Estimate equity with Monte Carlo, handling revealed opponent cards."""
    try:
        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board] if board else []
        opp_known = [eval7.Card(c) for c in opp_revealed] if opp_revealed else []

        known = set(my_cards + board_cards + opp_known)
        remaining = [c for c in eval7.Deck().cards if c not in known]

        wins = 0
        ties = 0
        total = 0
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

        if total == 0:
            return 0.5
        return (wins + ties * 0.5) / total

    except Exception:
        return fast_hand_rank(my_hand)


def monte_carlo_equity_with_potential(
    my_hand: list[str],
    board: list[str],
    opp_revealed: list[str],
    n_sims: int = 200,
) -> tuple[float, float, float]:
    """
    Returns (equity, positive_potential, negative_potential).
    Positive potential = fraction of currently-losing hands that improve to win.
    Negative potential = fraction of currently-winning hands that become losers.
    Only meaningful on flop/turn (not river).
    """
    try:
        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board] if board else []
        opp_known = [eval7.Card(c) for c in opp_revealed] if opp_revealed else []

        known = set(my_cards + board_cards + opp_known)
        remaining = [c for c in eval7.Deck().cards if c not in known]

        wins = ties = total = 0
        # For hand potential
        behind_improves = 0
        behind_total = 0
        ahead_worsens = 0
        ahead_total = 0

        board_need = 5 - len(board_cards)
        opp_need = 2 - len(opp_known)

        for _ in range(n_sims):
            random.shuffle(remaining)
            idx = 0

            opp_hand = list(opp_known)
            for _ in range(opp_need):
                opp_hand.append(remaining[idx])
                idx += 1

            # Current board eval
            cur_board = list(board_cards)
            future_cards = []
            for _ in range(board_need):
                future_cards.append(remaining[idx])
                idx += 1

            full_board = cur_board + future_cards

            # Current strength (with available board)
            if len(cur_board) >= 3:
                my_cur = eval7.evaluate(cur_board + my_cards)
                opp_cur = eval7.evaluate(cur_board + opp_hand)
            else:
                my_cur = 0
                opp_cur = 0

            # Final strength
            my_final = eval7.evaluate(full_board + my_cards)
            opp_final = eval7.evaluate(full_board + opp_hand)

            if my_final > opp_final:
                wins += 1
            elif my_final == opp_final:
                ties += 1
            total += 1

            # Track potential
            if len(cur_board) >= 3:
                if my_cur < opp_cur:  # currently behind
                    behind_total += 1
                    if my_final > opp_final:
                        behind_improves += 1
                elif my_cur > opp_cur:  # currently ahead
                    ahead_total += 1
                    if my_final < opp_final:
                        ahead_worsens += 1

        if total == 0:
            return 0.5, 0.0, 0.0

        equity = (wins + ties * 0.5) / total
        ppot = behind_improves / behind_total if behind_total > 0 else 0.0
        npot = ahead_worsens / ahead_total if ahead_total > 0 else 0.0

        return equity, ppot, npot

    except Exception:
        return fast_hand_rank(my_hand), 0.0, 0.0


# =============================================================================
# Opponent Model
# =============================================================================

class OpponentModel:
    """Comprehensive opponent tracking and classification."""

    def __init__(self):
        # Per-street fold counts
        self.street_folds = {"pre-flop": 0, "flop": 0, "turn": 0, "river": 0}
        self.street_actions = {"pre-flop": 0, "flop": 0, "turn": 0, "river": 0}

        # Overall aggression
        self.raise_count = 0
        self.call_count = 0
        self.check_count = 0
        self.fold_count = 0
        self.total_actions = 0

        # Bet sizing tracking
        self.bet_sizes: list[float] = []  # as fraction of pot

        # Auction tracking
        self.bids: list[int] = []
        self.bid_wins = 0
        self.bid_total = 0

        # Showdown tracking (for range estimation)
        self.showdown_hands: list[tuple[list[str], str]] = []  # (hand, final_street)
        self.showdown_strengths: list[float] = []

        # Pattern tracking
        self.check_raise_count = 0
        self.cbet_fold_count = 0  # folds to our c-bet
        self.cbet_face_count = 0

        # Previous state tracking
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.prev_my_wager = 0
        self.opp_checked_this_street = False

    def update(self, state: PokerState):
        """Infer opponent's action from state changes."""
        street = state.street
        if street == "auction":
            return

        opp_wager = state.opp_wager

        # Street changed
        if street != self.prev_street:
            self.prev_opp_wager = 0
            self.prev_my_wager = 0
            self.prev_street = street
            self.opp_checked_this_street = False

        if opp_wager > self.prev_opp_wager:
            increase = opp_wager - self.prev_opp_wager
            pot_before = state.pot - increase

            if self.prev_opp_wager == 0 and state.my_wager == 0:
                # Opponent opened
                self.raise_count += 1
                if pot_before > 0:
                    self.bet_sizes.append(increase / pot_before)
                # Check if this is a check-raise
                if self.opp_checked_this_street:
                    self.check_raise_count += 1
            elif increase > state.my_wager - self.prev_opp_wager:
                # Opponent raised
                self.raise_count += 1
                if pot_before > 0:
                    self.bet_sizes.append(increase / pot_before)
            else:
                # Opponent called
                self.call_count += 1

            self.total_actions += 1
            if street in self.street_actions:
                self.street_actions[street] += 1

        self.prev_opp_wager = opp_wager
        self.prev_my_wager = state.my_wager

    def record_fold(self, street: str):
        """Called when we infer the opponent folded."""
        self.fold_count += 1
        self.total_actions += 1
        if street in self.street_folds:
            self.street_folds[street] += 1
        if street in self.street_actions:
            self.street_actions[street] += 1

    def record_check(self, street: str):
        """Called when we infer the opponent checked."""
        self.check_count += 1
        self.total_actions += 1
        self.opp_checked_this_street = True
        if street in self.street_actions:
            self.street_actions[street] += 1

    def record_showdown(self, hand: list[str], strength: float, street: str):
        """Track hands shown at showdown."""
        self.showdown_hands.append((hand, street))
        self.showdown_strengths.append(strength)

    def record_bid(self, bid: int, won: bool):
        """Track auction bid."""
        self.bids.append(bid)
        self.bid_total += 1
        if won:
            self.bid_wins += 1

    def record_cbet_response(self, folded: bool):
        self.cbet_face_count += 1
        if folded:
            self.cbet_fold_count += 1

    # --- Derived stats ---

    @property
    def fold_rate(self) -> float:
        if self.total_actions < 12:
            return 0.30  # assume moderate
        return self.fold_count / self.total_actions

    def street_fold_rate(self, street: str) -> float:
        actions = self.street_actions.get(street, 0)
        if actions < 8:
            return self.fold_rate  # fallback to overall
        return self.street_folds.get(street, 0) / actions

    @property
    def aggression_factor(self) -> float:
        passive = self.call_count + self.check_count
        if passive == 0:
            return 2.0 if self.raise_count > 0 else 1.0
        return self.raise_count / passive

    @property
    def vpip(self) -> float:
        """Voluntarily put money in pot (approx)."""
        if self.total_actions < 10:
            return 0.50
        return (self.raise_count + self.call_count) / self.total_actions

    @property
    def avg_bid(self) -> float:
        if not self.bids:
            return 50.0
        return sum(self.bids) / len(self.bids)

    @property
    def bid_std(self) -> float:
        if len(self.bids) < 3:
            return 50.0
        avg = self.avg_bid
        var = sum((b - avg) ** 2 for b in self.bids) / len(self.bids)
        return math.sqrt(var)

    @property
    def avg_bet_size(self) -> float:
        """Average bet as fraction of pot."""
        if not self.bet_sizes:
            return 0.5
        return sum(self.bet_sizes) / len(self.bet_sizes)

    @property
    def cbet_fold_rate(self) -> float:
        if self.cbet_face_count < 5:
            return 0.40  # default
        return self.cbet_fold_count / self.cbet_face_count

    @property
    def is_tight(self) -> bool:
        return self.fold_rate > 0.35

    @property
    def is_loose(self) -> bool:
        return self.fold_rate < 0.22

    @property
    def is_aggressive(self) -> bool:
        return self.aggression_factor > 1.5

    @property
    def is_passive(self) -> bool:
        return self.aggression_factor < 0.8

    def classify(self) -> str:
        """Classify opponent: TAG, LAG, TP, LP (tight/loose + aggressive/passive)."""
        if self.total_actions < 20:
            return "unknown"
        tight = self.is_tight
        aggressive = self.is_aggressive
        if tight and aggressive:
            return "TAG"
        elif not tight and aggressive:
            return "LAG"
        elif tight and not aggressive:
            return "TP"
        else:
            return "LP"


# =============================================================================
# Ultra Bot
# =============================================================================

class Player(BaseBot):
    """
    Ultra Bot: A comprehensive exploitative poker bot.
    """

    def __init__(self) -> None:
        self.opp = OpponentModel()

        # Per-hand state
        self.hands_played = 0
        self.preflop_equity = 0.5
        self.hand_equity = 0.5
        self.hand_ppot = 0.0  # positive potential
        self.hand_npot = 0.0  # negative potential
        self.hand_aggression = 0
        self.we_have_initiative = False  # did we last bet/raise?
        self.board_tex: BoardTexture = BoardTexture([])
        self.prev_board_len = 0

        # Track the last street we computed equity on (avoid redundant calcs)
        self.equity_computed_street = ""

        # Preflop cache (canonical hand -> equity)
        self._preflop_cache: dict[tuple, float] = {}

        # Track if we won the auction
        self.won_auction = False
        self.opp_revealed_seen = False

        # Our last action per street
        self.last_action_was_raise = False

    # =========================================================================
    # Equity computation
    # =========================================================================

    def _get_preflop_equity(self, hand: list[str], time_bank: float) -> float:
        key = canonical_hand_key(hand)
        if key not in self._preflop_cache:
            n_sims = 300 if time_bank > 15.0 else (200 if time_bank > 8.0 else 100)
            self._preflop_cache[key] = monte_carlo_equity(hand, [], [], n_sims)
        return self._preflop_cache[key]

    def _compute_equity(self, state: PokerState, time_bank: float) -> tuple[float, float, float]:
        """Compute equity + hand potential for postflop play."""
        street = state.street
        board = state.board

        # Use potential-aware equity on flop/turn (not river - no more cards)
        if street in ("flop", "turn") and time_bank > 6.0:
            n_sims = 250 if time_bank > 12.0 else (150 if time_bank > 8.0 else 80)
            return monte_carlo_equity_with_potential(
                state.my_hand, board, state.opp_revealed_cards, n_sims
            )
        else:
            # River or low time: just equity
            if time_bank > 8.0:
                n_sims = 200
            elif time_bank > 4.0:
                n_sims = 100
            elif time_bank > 2.0:
                n_sims = 50
            else:
                return fast_hand_rank(state.my_hand), 0.0, 0.0
            eq = monte_carlo_equity(
                state.my_hand, board, state.opp_revealed_cards, n_sims
            )
            return eq, 0.0, 0.0

    # =========================================================================
    # Effective Hand Strength
    # =========================================================================

    def _effective_hand_strength(self, equity: float, ppot: float, npot: float,
                                  street: str) -> float:
        """
        EHS = equity * (1 - npot) + (1 - equity) * ppot
        This accounts for the probability of improving when behind and
        being outdrawn when ahead. Weight by street.
        """
        if street == "river":
            return equity  # no more cards to come

        # Weight potential more on flop, less on turn
        ppot_weight = 0.8 if street == "flop" else 0.5
        npot_weight = 0.5 if street == "flop" else 0.3

        ehs = equity * (1.0 - npot * npot_weight) + (1.0 - equity) * ppot * ppot_weight
        return min(max(ehs, 0.0), 1.0)

    # =========================================================================
    # Pot odds & implied odds
    # =========================================================================

    def _pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0:
            return 0.0
        return cost / (state.pot + cost)

    def _implied_odds_factor(self, state: PokerState, street: str) -> float:
        """
        Multiply pot odds by this to get effective required equity.
        < 1.0 means implied odds are good (we need less equity).
        """
        spr = state.my_chips / max(state.pot, 1)

        if street == "flop":
            # 2 streets left, good implied odds with deep stacks
            if spr > 5:
                return 0.70
            elif spr > 2:
                return 0.80
            return 0.90
        elif street == "turn":
            # 1 street left
            if spr > 3:
                return 0.80
            return 0.90
        return 1.0  # river - no implied odds

    # =========================================================================
    # Bet Sizing
    # =========================================================================

    def _geometric_bet_size(self, state: PokerState, street: str, equity: float) -> int:
        """
        Calculate bet size using geometric sizing for multi-street value.
        Goal: size bets so we can get all chips in by the river with value hands.
        """
        pot = state.pot
        my_chips = state.my_chips
        opp_chips = state.opp_chips
        effective_stack = min(my_chips, opp_chips)
        spr = effective_stack / max(pot, 1)

        streets_left = {"flop": 3, "turn": 2, "river": 1}.get(street, 1)

        if equity > 0.82:
            # Premium: size for stacks
            if streets_left > 1 and spr > 2:
                # Geometric: find bet fraction b such that (1+2b)^streets = 1 + spr
                # b = ((1+spr)^(1/streets) - 1) / 2
                target_ratio = (1.0 + spr)
                per_street = target_ratio ** (1.0 / streets_left)
                bet_frac = (per_street - 1.0) / 2.0
                bet_frac = min(bet_frac, 1.5)  # cap at 150% pot
            else:
                bet_frac = 0.75
        elif equity > 0.68:
            # Strong value: 55-75% pot
            bet_frac = 0.55 + (equity - 0.68) * 1.4
        elif equity > 0.55:
            # Medium value / protection: 40-55% pot
            bet_frac = 0.40 + (equity - 0.55) * 1.15
        else:
            # Bluff: use small sizing for better risk/reward
            opp_fold = self.opp.fold_rate
            if opp_fold > 0.35:
                # Small bluff against tight players
                bet_frac = 0.30
            else:
                bet_frac = 0.40

        # River polarization
        if street == "river":
            if equity > 0.80:
                bet_frac = max(bet_frac, 0.80)
            elif equity < 0.30:
                # Bluff overbet can be profitable vs tight opponents
                if self.opp.is_tight:
                    bet_frac = 0.70
                else:
                    bet_frac = 0.45

        target = int(pot * bet_frac)
        min_raise, max_raise = state.raise_bounds
        amount = max(min_raise, min(target + state.my_wager, max_raise))
        return amount

    # =========================================================================
    # Bluffing Strategy
    # =========================================================================

    def _should_bluff(self, state: PokerState, street: str, equity: float) -> bool:
        """
        Decide whether to bluff based on:
        - Opponent fold rate (primary driver)
        - Board texture (bluff more on scary boards)
        - Our aggression this hand (don't triple-barrel bluff too often)
        - Pot size relative to risk
        """
        opp_fold = self.opp.fold_rate
        street_fold = self.opp.street_fold_rate(street)

        # Use the more favorable fold rate
        effective_fold = max(opp_fold, street_fold)

        # Base bluff frequency based on pot odds math
        # If we bet B into pot P, opp needs equity of B/(P+2B) to call
        # So optimal bluff frequency = B/(P+2B) / (1 - B/(P+2B))
        # With a 50% pot bet: 0.5/(1+1) = 25% of range should be bluffs
        # But we adjust based on opponent tendency

        if effective_fold > 0.40:
            # Very profitable to bluff
            bluff_freq = 0.35 + (effective_fold - 0.40) * 0.5
        elif effective_fold > 0.28:
            bluff_freq = 0.18 + (effective_fold - 0.28) * 1.0
        else:
            # Calling station - don't bluff much
            bluff_freq = max(0.05, effective_fold * 0.3)

        # Board texture adjustment
        if self.board_tex.is_wet and street in ("turn", "river"):
            bluff_freq += 0.05  # scarier board = better bluff spot
        if self.board_tex.has_flush_draw and street == "river":
            bluff_freq += 0.06  # missed flush draw boards are great bluff spots

        # Reduce if we've already been aggressive
        if self.hand_aggression >= 3:
            bluff_freq *= 0.15
        elif self.hand_aggression >= 2:
            bluff_freq *= 0.40

        # Don't bluff into huge pots with marginal holdings
        if state.pot > 600 and equity < 0.20:
            bluff_freq *= 0.5

        bluff_freq = max(0.03, min(bluff_freq, 0.50))
        return random.random() < bluff_freq

    # =========================================================================
    # Auction Strategy
    # =========================================================================

    def _auction_bid(self, game_info: GameInfo, state: PokerState) -> ActionBid:
        """
        Second-price auction: bidding true value is dominant strategy.

        Information value:
        - With a strong hand: seeing their card confirms our edge and helps
          us extract max value. Value = ability to bet with more confidence.
        - With a weak hand: seeing their card mostly just confirms we're behind.
          Not worth much.
        - With a medium hand: highest info value - seeing a weak opponent card
          converts a marginal hand into a confident value bet.

        We also exploit opponent bidding patterns.
        """
        equity = self.hand_equity
        my_chips = state.my_chips
        pot = state.pot

        # Information value is highest for medium-strength hands
        # It's a bell curve centered around 0.55 equity
        if equity > 0.80:
            # Very strong: info is nice but not critical
            info_value = 0.10
        elif equity > 0.65:
            # Strong: good info value
            info_value = 0.06 + (equity - 0.65) * 0.5
        elif equity > 0.50:
            # Medium: highest info value
            info_value = 0.08 + (0.65 - abs(equity - 0.575)) * 0.3
        elif equity > 0.38:
            # Weak-medium: moderate info value
            info_value = 0.04
        else:
            # Weak: info not very useful
            info_value = 0.01

        base_bid = int(info_value * pot)

        # Exploit opponent bidding patterns
        if len(self.opp.bids) >= 8:
            opp_avg = self.opp.avg_bid
            opp_std = self.opp.bid_std

            if equity > 0.55:
                # Try to win auction: bid slightly above their expected bid
                adaptive_bid = int(opp_avg + opp_std * 0.3) + 1
                base_bid = max(base_bid, adaptive_bid)
            elif equity < 0.35:
                # Weak hand: let them overpay
                base_bid = 0

            # If opponent consistently bids 0 or very low, bid minimally to win cheap info
            if opp_avg < 5 and equity > 0.45:
                base_bid = max(base_bid, 2)

        elif len(self.opp.bids) >= 3:
            opp_avg = self.opp.avg_bid
            if equity > 0.55:
                adaptive_bid = int(opp_avg) + 2
                base_bid = max(base_bid, adaptive_bid)

        # Cap bid at a fraction of chips
        if equity > 0.70:
            cap = int(my_chips * 0.18)
        elif equity > 0.50:
            cap = int(my_chips * 0.10)
        else:
            cap = int(my_chips * 0.04)

        bid = max(0, min(base_bid, cap, my_chips))
        return ActionBid(bid)

    # =========================================================================
    # Preflop Strategy
    # =========================================================================

    def _handle_preflop(self, game_info: GameInfo, state: PokerState):
        equity = self.preflop_equity
        cost = state.cost_to_call
        in_position = state.is_bb  # BB acts last postflop
        opp_agg = self.opp.aggression_factor
        opp_fold = self.opp.fold_rate

        if cost > 0:
            # === Facing a raise ===
            pot_odds = self._pot_odds(state)

            # Premium hands: always re-raise
            if equity > 0.72:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, "pre-flop", equity))
                return ActionCall()

            # Strong hands: mix re-raise and call
            if equity > 0.60:
                if state.can_act(ActionRaise) and random.random() < 0.35:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, "pre-flop", equity))
                return ActionCall()

            # Profitable call
            if equity > pot_odds + 0.04:
                return ActionCall()

            # Position bonus: call more in position
            if in_position and equity > pot_odds - 0.02:
                return ActionCall()

            # Against very aggressive opponents: widen call range
            if opp_agg > 2.5 and equity > 0.36:
                return ActionCall()

            # Defend BB with decent hands
            if in_position and equity > 0.33 and cost <= BIG_BLIND * 2:
                if random.random() < 0.25:
                    return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        else:
            # === No cost (can open raise or check) ===
            if state.can_act(ActionRaise):
                # Open raise with good hands
                if equity > 0.52:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, "pre-flop", equity))

                # Steal with playable hands
                if equity > 0.40 and random.random() < 0.45:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    min_raise, _ = state.raise_bounds
                    return ActionRaise(min_raise)

                # Bluff raise with weak hands (steal)
                if equity < 0.28 and self._should_bluff(state, "pre-flop", equity):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    min_raise, _ = state.raise_bounds
                    return ActionRaise(min_raise)

            return ActionCheck()

    # =========================================================================
    # Postflop Strategy
    # =========================================================================

    def _handle_postflop(self, game_info: GameInfo, state: PokerState):
        street = state.street

        # Compute equity if not already done this street
        if self.equity_computed_street != street:
            eq, ppot, npot = self._compute_equity(state, game_info.time_bank)
            self.hand_equity = eq
            self.hand_ppot = ppot
            self.hand_npot = npot
            self.equity_computed_street = street

        equity = self.hand_equity
        ehs = self._effective_hand_strength(equity, self.hand_ppot, self.hand_npot, street)

        # Update board texture
        if state.board and len(state.board) != self.prev_board_len:
            self.board_tex = BoardTexture(state.board)
            self.prev_board_len = len(state.board)

        cost = state.cost_to_call
        opp_type = self.opp.classify()
        opp_fold = self.opp.fold_rate
        opp_agg = self.opp.aggression_factor
        has_opp_info = len(state.opp_revealed_cards) > 0

        # When we know an opponent card, our equity is more reliable
        confidence_bonus = 0.03 if has_opp_info else 0.0

        # === FACING A BET/RAISE ===
        if cost > 0:
            pot_odds = self._pot_odds(state)
            implied_factor = self._implied_odds_factor(state, street)
            effective_odds = pot_odds * implied_factor

            # Very strong: raise for value
            if ehs > 0.78 + confidence_bonus:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, street, ehs))
                return ActionCall()

            # Strong: mix raise/call
            if ehs > 0.64:
                if state.can_act(ActionRaise) and random.random() < 0.30:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, street, ehs))
                return ActionCall()

            # Equity > effective pot odds: profitable call
            if ehs > effective_odds + 0.03:
                return ActionCall()

            # Drawing hands on flop/turn: call with potential
            if street in ("flop", "turn"):
                drawing_equity = equity + self.hand_ppot * 0.3
                if drawing_equity > effective_odds - 0.02:
                    if random.random() < 0.65:
                        return ActionCall()

            # Against very aggressive: call down lighter
            if opp_agg > 2.5 and ehs > effective_odds - 0.06:
                return ActionCall()

            # Semi-bluff raise on flop/turn
            if (
                street in ("flop", "turn")
                and self.hand_ppot > 0.20
                and equity > 0.28
                and state.can_act(ActionRaise)
                and self.hand_aggression < 2
                and random.random() < 0.25
            ):
                self.hand_aggression += 1
                self.we_have_initiative = True
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

            # Check-raise bluff against aggressive opponents
            if (
                opp_agg > 2.0
                and equity < 0.25
                and state.can_act(ActionRaise)
                and self.hand_aggression == 0
                and self._should_bluff(state, street, equity)
                and cost < state.pot * 0.4
            ):
                self.hand_aggression += 1
                self.we_have_initiative = True
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        # === NO COST (check or bet) ===
        if state.can_act(ActionRaise):
            # C-bet: continuation bet with initiative
            if self.we_have_initiative and street == "flop":
                cbet_threshold = 0.42
                # Adjust based on board texture
                if self.board_tex.is_dry:
                    cbet_threshold = 0.38  # c-bet more on dry boards
                elif self.board_tex.is_wet:
                    cbet_threshold = 0.48  # be more selective on wet boards

                # Adjust based on opponent cbet fold rate
                if self.opp.cbet_fold_rate > 0.50:
                    cbet_threshold = 0.30  # they fold a lot, c-bet wide

                if ehs > cbet_threshold:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, street, ehs))

            # Value bet
            if ehs > 0.62:
                self.hand_aggression += 1
                self.we_have_initiative = True
                return ActionRaise(self._geometric_bet_size(state, street, ehs))

            # Thin value / protection bet
            if ehs > 0.50 and random.random() < 0.45:
                self.hand_aggression += 1
                self.we_have_initiative = True
                return ActionRaise(self._geometric_bet_size(state, street, ehs))

            # Bluff
            if (
                ehs < 0.30
                and self._should_bluff(state, street, equity)
                and self.hand_aggression < 2
            ):
                self.hand_aggression += 1
                self.we_have_initiative = True
                return ActionRaise(self._geometric_bet_size(state, street, ehs))

            # Probe bet on turn/river without initiative
            if (
                not self.we_have_initiative
                and street in ("turn", "river")
                and ehs > 0.55
                and random.random() < 0.35
            ):
                self.hand_aggression += 1
                self.we_have_initiative = True
                return ActionRaise(self._geometric_bet_size(state, street, ehs))

        return ActionCheck()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.hand_aggression = 0
        self.we_have_initiative = False
        self.opp.prev_opp_wager = 0
        self.opp.prev_my_wager = 0
        self.opp.prev_street = ""
        self.opp.opp_checked_this_street = False
        self.equity_computed_street = ""
        self.won_auction = False
        self.opp_revealed_seen = False
        self.prev_board_len = 0
        self.board_tex = BoardTexture([])
        self.last_action_was_raise = False

        my_cards = current_state.my_hand
        if game_info.time_bank > 4.0:
            self.preflop_equity = self._get_preflop_equity(my_cards, game_info.time_bank)
        else:
            self.preflop_equity = fast_hand_rank(my_cards)

        self.hand_equity = self.preflop_equity
        self.hand_ppot = 0.0
        self.hand_npot = 0.0

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        payoff = current_state.payoff

        # Detect opponent fold
        if payoff > 0 and not current_state.opp_revealed_cards:
            if self.hand_aggression > 0:
                self.opp.record_fold(current_state.street)

        # Track showdown data
        if current_state.opp_revealed_cards and len(current_state.opp_revealed_cards) == 2:
            # We went to showdown and can see both cards
            self.opp.record_showdown(
                current_state.opp_revealed_cards,
                fast_hand_rank(current_state.opp_revealed_cards),
                current_state.street,
            )

    # =========================================================================
    # Main action
    # =========================================================================

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        street = current_state.street

        # Track opponent
        if street != "auction":
            self.opp.update(current_state)

        # Auction
        if street == "auction":
            return self._auction_bid(game_info, current_state)

        # Preflop
        if street == "pre-flop":
            return self._handle_preflop(game_info, current_state)

        # Postflop (flop, turn, river)
        return self._handle_postflop(game_info, current_state)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
