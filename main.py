"""
Advanced poker bot for IIT Pokerbots 2026.

Strategy overview:
- Monte Carlo equity estimation using eval7 for hand strength
- Position-aware preflop ranges with raise/call/fold thresholds
- Auction bidding calibrated to hand equity and information value
- Postflop play using pot odds vs equity with aggression scaling
- Opponent modeling: tracks fold frequency, aggression, and auction bids
- Time-aware: uses cheap heuristics when time bank is low
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random

# Constants
NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10

# Card rank values for quick heuristic evaluation
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def card_str_to_eval7(card_str: str) -> eval7.Card:
    """Convert engine card string like 'Ah' to eval7.Card."""
    return eval7.Card(card_str)


def fast_hand_rank(cards: list[str]) -> float:
    """
    Quick heuristic hand strength for 2-card hands.
    Returns a value from 0 to 1 representing relative preflop strength.
    """
    if len(cards) != 2:
        return 0.5

    r1 = RANK_VALUES.get(cards[0][0], 0)
    r2 = RANK_VALUES.get(cards[1][0], 0)
    suited = cards[0][1] == cards[1][1]
    high = max(r1, r2)
    low = min(r1, r2)

    # Pocket pairs
    if r1 == r2:
        return 0.5 + (high - 2) * 0.038  # AA ~0.96, 22 ~0.50

    # Suited bonus
    suit_bonus = 0.03 if suited else 0.0

    # Connectedness bonus
    gap = high - low
    connect_bonus = max(0, 0.02 - gap * 0.003)

    # High card value
    score = (high + low - 4) / 24.0 + suit_bonus + connect_bonus

    # Premium hands boost
    if high == 14:  # Ace-x
        score += 0.08
    if high >= 13 and low >= 11:  # Broadway
        score += 0.05

    return min(max(score, 0.0), 1.0)


def monte_carlo_equity(
    my_hand: list[str],
    board: list[str],
    opp_revealed: list[str],
    num_simulations: int = 150,
) -> float:
    """
    Estimate hand equity via Monte Carlo simulation.
    Deals random opponent hands and remaining board cards.
    """
    try:
        my_cards = [card_str_to_eval7(c) for c in my_hand]
        board_cards = [card_str_to_eval7(c) for c in board] if board else []
        opp_known = [card_str_to_eval7(c) for c in opp_revealed] if opp_revealed else []

        # All known cards (to exclude from deck)
        known = set(my_cards + board_cards + opp_known)

        # Build remaining deck
        full_deck = eval7.Deck()
        remaining = [c for c in full_deck.cards if c not in known]

        wins = 0
        ties = 0
        total = 0

        cards_needed_for_board = 5 - len(board_cards)
        opp_cards_needed = 2 - len(opp_known)

        for _ in range(num_simulations):
            random.shuffle(remaining)

            idx = 0
            # Deal opponent's unknown cards
            opp_hand = list(opp_known)
            for _ in range(opp_cards_needed):
                opp_hand.append(remaining[idx])
                idx += 1

            # Deal remaining board
            sim_board = list(board_cards)
            for _ in range(cards_needed_for_board):
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


class Player(BaseBot):
    """
    An advanced poker bot with:
    - Monte Carlo equity estimation
    - Adaptive auction bidding
    - Opponent modeling
    - Position-aware play
    - Time management
    """

    def __init__(self) -> None:
        # Opponent modeling
        self.opp_fold_count = 0
        self.opp_raise_count = 0
        self.opp_call_count = 0
        self.opp_check_count = 0
        self.opp_total_actions = 0
        self.opp_bids: list[int] = []
        self.opp_showdown_count = 0

        # Per-hand tracking
        self.hands_played = 0
        self.hand_equity = 0.5
        self.preflop_equity = 0.5
        self.hand_aggression = 0
        self.prev_opp_wager = 0  # track opponent's wager from last action
        self.prev_street = ""
        self.opp_acted_this_street = False

        # Preflop equity cache (canonical hand -> equity)
        self._preflop_cache: dict[tuple, float] = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_preflop_equity(self, hand: list[str], time_bank: float) -> float:
        """Get preflop equity with caching by canonical hand representation."""
        r1 = hand[0][0]
        r2 = hand[1][0]
        suited = hand[0][1] == hand[1][1]

        ranks = tuple(sorted([r1, r2], key=lambda x: RANK_VALUES[x], reverse=True))
        key = (ranks[0], ranks[1], suited)

        if key not in self._preflop_cache:
            n_sims = 200 if time_bank > 10.0 else 100
            self._preflop_cache[key] = monte_carlo_equity(
                hand, [], [], num_simulations=n_sims
            )

        return self._preflop_cache[key]

    def _get_opp_fold_rate(self) -> float:
        if self.opp_total_actions < 15:
            return 0.30
        return self.opp_fold_count / self.opp_total_actions

    def _get_opp_aggression_factor(self) -> float:
        passive = self.opp_call_count + self.opp_check_count
        if passive == 0:
            return 2.0 if self.opp_raise_count > 0 else 1.0
        return self.opp_raise_count / passive

    def _get_opp_avg_bid(self) -> float:
        if not self.opp_bids:
            return 50.0
        return sum(self.opp_bids) / len(self.opp_bids)

    def _get_pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0:
            return 0.0
        return cost / (state.pot + cost)

    def _infer_opp_action(self, state: PokerState) -> None:
        """
        Infer what opponent did by comparing current wager/pot to previous state.
        Called at the start of get_move to track opponent behavior.
        """
        street = state.street
        opp_wager = state.opp_wager

        # Street changed - reset tracking
        if street != self.prev_street:
            self.prev_opp_wager = 0
            self.prev_street = street
            self.opp_acted_this_street = False

        # If opponent has put in more money than before, they raised/bet
        if opp_wager > self.prev_opp_wager:
            wager_increase = opp_wager - self.prev_opp_wager
            if self.prev_opp_wager == 0 and state.my_wager == 0:
                # Opponent opened with a bet
                self.opp_raise_count += 1
            elif wager_increase > state.my_wager - self.prev_opp_wager:
                # Opponent raised
                self.opp_raise_count += 1
            else:
                # Opponent called
                self.opp_call_count += 1
            self.opp_total_actions += 1
            self.opp_acted_this_street = True

        self.prev_opp_wager = opp_wager

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.hands_played += 1
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.opp_acted_this_street = False

        my_cards = current_state.my_hand
        if game_info.time_bank > 5.0:
            self.preflop_equity = self._get_preflop_equity(
                my_cards, game_info.time_bank
            )
        else:
            self.preflop_equity = fast_hand_rank(my_cards)

        self.hand_equity = self.preflop_equity

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        # If opponent folded, count it
        if current_state.payoff > 0 and not current_state.opp_revealed_cards:
            # We won without showdown - opponent folded at some point
            # Only count if we were aggressive (we bet/raised and they folded)
            if self.hand_aggression > 0:
                self.opp_fold_count += 1
                self.opp_total_actions += 1

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def _should_bluff(self, state: PokerState) -> bool:
        opp_fold_rate = self._get_opp_fold_rate()

        # Base bluff frequency ~25%, more against foldy opponents
        bluff_freq = 0.20 + (opp_fold_rate - 0.30) * 0.5
        bluff_freq = max(0.08, min(0.45, bluff_freq))

        # Reduce bluffing if we've already been aggressive this hand
        if self.hand_aggression >= 2:
            bluff_freq *= 0.3

        return random.random() < bluff_freq

    def _get_raise_amount(
        self,
        state: PokerState,
        equity: float,
        street: str,
    ) -> int:
        min_raise, max_raise = state.raise_bounds
        pot = state.pot

        if equity > 0.85:
            target = int(pot * 0.85)
        elif equity > 0.7:
            target = int(pot * 0.65)
        elif equity > 0.55:
            target = int(pot * 0.45)
        else:
            # Bluff sizing: use ~33-40% pot
            target = int(pot * 0.35)

        # River: polarize
        if street == "river":
            if equity > 0.8:
                target = int(pot * 1.0)
            elif equity < 0.35:
                target = int(pot * 0.55)

        amount = max(min_raise, min(target + state.my_wager, max_raise))
        return amount

    def _auction_bid(
        self,
        game_info: GameInfo,
        state: PokerState,
    ) -> ActionBid:
        """
        Auction strategy:
        - The winner pays the LOSER's bid and sees one of loser's cards
        - So we want to bid just enough to win but pay as little as possible
        - With strong hands, seeing opponent's card confirms our advantage
        - With weak hands, bid 0 to avoid paying
        - On ties, both pay their own bid and both see a card
        """
        equity = self.hand_equity
        my_chips = state.my_chips
        pot = state.pot

        if equity > 0.75:
            # Strong hand: seeing their card is valuable
            # Bid moderately - we'll pay their bid, not ours
            bid_frac = 0.12 + (equity - 0.75) * 0.4
        elif equity > 0.55:
            # Medium hand: info is somewhat useful
            bid_frac = 0.04 + (equity - 0.55) * 0.2
        elif equity > 0.40:
            # Marginal: small bid
            bid_frac = 0.01
        else:
            # Weak: don't bid
            bid_frac = 0.0

        base_bid = int(bid_frac * pot)

        # Adaptive: try to outbid opponent's average by a small margin
        if len(self.opp_bids) >= 5:
            opp_avg = self._get_opp_avg_bid()
            if equity > 0.6:
                # Outbid them slightly
                adaptive_bid = int(opp_avg) + 2
                base_bid = max(base_bid, adaptive_bid)
            elif equity < 0.40:
                # Weak hand: keep bid very low
                base_bid = min(base_bid, 1)

        # Cap at a fraction of our chips
        if equity > 0.75:
            cap = int(my_chips * 0.20)
        else:
            cap = int(my_chips * 0.10)

        bid = max(0, min(base_bid, cap, my_chips))
        return ActionBid(bid)

    def _handle_preflop(
        self,
        game_info: GameInfo,
        state: PokerState,
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise:
        equity = self.preflop_equity
        cost = state.cost_to_call
        in_position = state.is_bb

        # Facing a raise
        if cost > 0:
            pot_odds = self._get_pot_odds(state)
            opp_agg = self._get_opp_aggression_factor()

            # Premium: re-raise
            if equity > 0.70 and state.can_act(ActionRaise):
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, "pre-flop"))

            # Good: call or re-raise
            if equity > 0.58 and state.can_act(ActionRaise) and random.random() < 0.3:
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, "pre-flop"))

            # Profitable call
            if equity > pot_odds + 0.05:
                return ActionCall()

            # In position: call more liberally
            if in_position and equity > pot_odds - 0.03:
                return ActionCall()

            # Against very aggressive opponents, widen calling range
            if opp_agg > 2.0 and equity > 0.38:
                return ActionCall()

            # Defend BB with decent hands
            if in_position and equity > 0.35 and random.random() < 0.20:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        # No cost (can check or open-raise)
        if state.can_act(ActionRaise):
            if equity > 0.55:
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, "pre-flop"))

            if equity > 0.42 and random.random() < 0.40:
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

            # Bluff with some weak hands
            if equity < 0.30 and self._should_bluff(state):
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

        return ActionCheck()

    def _compute_equity(
        self,
        state: PokerState,
        time_bank: float,
    ) -> float:
        """Compute equity with adaptive simulation count based on time."""
        if time_bank > 10.0:
            n_sims = 150
        elif time_bank > 5.0:
            n_sims = 80
        elif time_bank > 2.0:
            n_sims = 40
        else:
            # Very low time: use fast heuristic
            return fast_hand_rank(state.my_hand)

        return monte_carlo_equity(
            state.my_hand, state.board, state.opp_revealed_cards, n_sims
        )

    def _handle_postflop(
        self,
        game_info: GameInfo,
        state: PokerState,
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise:
        street = state.street
        equity = self._compute_equity(state, game_info.time_bank)
        self.hand_equity = equity
        cost = state.cost_to_call
        opp_agg = self._get_opp_aggression_factor()

        # When we know an opponent card, our equity estimate is more reliable
        has_opp_info = len(state.opp_revealed_cards) > 0

        # Facing a bet/raise
        if cost > 0:
            pot_odds = self._get_pot_odds(state)

            # Adjust for implied odds on non-river streets
            effective_odds = pot_odds
            if street in ("flop", "turn"):
                effective_odds = pot_odds * 0.82

            # Very strong: re-raise for value
            if equity > 0.78 and state.can_act(ActionRaise):
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, street))

            # Strong: raise or call
            if equity > 0.62 and state.can_act(ActionRaise):
                if random.random() < 0.35:
                    self.hand_aggression += 1
                    return ActionRaise(self._get_raise_amount(state, equity, street))
                return ActionCall()

            # Equity > pot odds: call
            if equity > effective_odds + 0.03:
                return ActionCall()

            # Drawing hands on flop/turn
            if street in ("flop", "turn") and equity > effective_odds - 0.04:
                if random.random() < 0.55:
                    return ActionCall()

            # Against very aggressive opponents, call down lighter
            if opp_agg > 2.5 and equity > effective_odds - 0.05:
                return ActionCall()

            # Semi-bluff raise
            if (
                street in ("flop", "turn")
                and equity > 0.32
                and state.can_act(ActionRaise)
                and self._should_bluff(state)
                and self.hand_aggression < 2
            ):
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()

        # No cost: check or bet
        if state.can_act(ActionRaise):
            # Value bet
            if equity > 0.62:
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, street))

            # Thin value / protection bet
            if equity > 0.50 and random.random() < 0.40:
                self.hand_aggression += 1
                return ActionRaise(self._get_raise_amount(state, equity, street))

            # Bluff
            if (
                equity < 0.32
                and self._should_bluff(state)
                and self.hand_aggression < 2
            ):
                self.hand_aggression += 1
                min_raise, _ = state.raise_bounds
                return ActionRaise(min_raise)

        return ActionCheck()

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:

        street = current_state.street

        # Track opponent behavior (skip for auction street)
        if street != "auction":
            self._infer_opp_action(current_state)

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
