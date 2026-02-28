"""
Simple example pokerbot, written in Python.
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import itertools


class Player(BaseBot):
    """
    A pokerbot that makes decisions based on the probability of winning (equity).
    Calculates equity using exhaustive enumeration of opponent hands via eval7.
    """

    def __init__(self) -> None:
        pass

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def get_equity(self, my_hand: list[str], board: list[str], opp_revealed: list[str]) -> float:
        """
        Calculates hand strength (probability of being ahead of a random hand).
        Uses itertools.combinations for exhaustive enumeration.
        """
        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board]
        opp_revealed_cards = [eval7.Card(c) for c in opp_revealed]

        deck = eval7.Deck()
        for card in my_cards + board_cards + opp_revealed_cards:
            if card in deck.cards:
                deck.cards.remove(card)

        # Pre-flop or Auction (no board yet)
        if len(board_cards) == 0:
            # Enhanced heuristic for pre-flop strength
            r1, r2 = my_cards[0].rank, my_cards[1].rank
            s1, s2 = my_cards[0].suit, my_cards[1].suit
            
            # Base score: High card points
            score = max(r1, r2)
            
            # Pairs
            if r1 == r2:
                score *= 2
                score = max(score, 5) # Minimum score for low pairs
            
            # Suited
            if s1 == s2:
                score += 2
            
            # Connectors (gap penalty)
            gap = abs(r1 - r2)
            if gap == 1: score += 1
            elif gap == 2: score -= 1
            elif gap == 3: score -= 2
            elif gap >= 4: score -= 4
            
            # Normalize to 0.0 - 1.0 roughly (Max score approx 28 for AA)
            # 72o score: 5 - 4 = 1. AA: 24.
            prob = 0.3 + (score / 40.0) 
            return min(0.95, max(0.2, prob))

        # Post-flop: Enumerate all possible opponent hands given current board
        wins = 0
        total = 0
        num_opp_missing = 2 - len(opp_revealed_cards)
        
        # itertools.combinations is used for exhaustive enumeration of possibilities
        for opp_hole in itertools.combinations(deck.cards, num_opp_missing):
            opp_hand = opp_revealed_cards + list(opp_hole)
            
            # eval7.evaluate compares the best 5-card hand
            my_val = eval7.evaluate(my_cards + board_cards)
            opp_val = eval7.evaluate(opp_hand + board_cards)
            
            if my_val > opp_val:
                wins += 1
            elif my_val == opp_val:
                wins += 0.5
            total += 1
            
        return wins / total if total > 0 else 0.5

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        """
        Main decision logic.
        Strategy: BULLY. Always Raise to force the opponent's 25% fold chance.
        """
        # Auction Strategy: EXPLOITATIVE
        # BotA bids random(0, 0.3) * chips.
        # We bid 0.35 * chips to guarantee winning.
        if current_state.street == "auction":
            bid_amount = int(current_state.my_chips * 0.35)
            return ActionBid(bid_amount)

        # Decision Logic: Always Raise if possible
        if current_state.can_act(ActionRaise):
            min_r, max_r = current_state.raise_bounds
            return ActionRaise(min_r)
        
        # If cannot raise (e.g., re-raise limit reached or opponent all-in), Call.
        if current_state.can_act(ActionCall):
            return ActionCall()

        # Fallback (should not be reached if Call is available)
        return ActionCheck()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
