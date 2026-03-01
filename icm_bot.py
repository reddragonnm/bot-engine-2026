"""
ICM-Enhanced Poker Bot v3 - UltraBot Crusher
Optimized to beat UltraBot specifically.
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random
import math

from icm import ICMCalculator, ICMBotHelper


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
        return 0.5 + (high - 2) * 0.042
    suit_bonus = 0.038 if suited else 0.0
    gap = high - low
    connect_bonus = max(0, 0.028 - gap * 0.004)
    score = (high + low - 4) / 24.0 + suit_bonus + connect_bonus
    if high == 14:
        score += 0.10
    if high >= 13 and low >= 11:
        score += 0.07
    if high >= 13 and low >= 10:
        score += 0.03
    return min(max(score, 0.0), 1.0)


def monte_carlo_equity(my_hand, board, opp_revealed, n_sims=150):
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


class ICMPlayerV3(BaseBot):
    def __init__(self):
        self.icm_helper = ICMBotHelper(prize_pool=100.0)
        self.icm = ICMCalculator(prize_pool=100.0)

        self.opp_fold_count = 0
        self.opp_raise_count = 0
        self.opp_call_count = 0
        self.opp_check_count = 0
        self.opp_total_actions = 0
        self.opp_bids = []
        self.opp_bet_sizes = []

        self.hands_played = 0
        self.hand_equity = 0.5
        self.preflop_equity = 0.5
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.we_have_initiative = False

        self._preflop_cache = {}

    def _opp_fold_rate(self):
        if self.opp_total_actions < 10:
            return 0.30
        return self.opp_fold_count / self.opp_total_actions

    def _opp_aggression(self):
        passive = self.opp_call_count + self.opp_check_count
        if passive == 0:
            return 2.0 if self.opp_raise_count > 0 else 1.0
        return self.opp_raise_count / max(passive, 1)

    def _opp_avg_bid(self):
        if not self.opp_bids:
            return 10.0
        return sum(self.opp_bids) / len(self.opp_bids)

    def _opp_bid_std(self):
        if len(self.opp_bids) < 3:
            return 8.0
        avg = self._opp_avg_bid()
        var = sum((b - avg) ** 2 for b in self.opp_bids) / len(self.opp_bids)
        return math.sqrt(var)

    def _infer_opp(self, state: PokerState):
        street = state.street
        ow = state.opp_wager
        pot = state.pot
        
        if street != self.prev_street:
            self.prev_opp_wager = 0
            self.prev_street = street
        if ow > self.prev_opp_wager:
            inc = ow - self.prev_opp_wager
            pot_before = pot - inc
            
            if self.prev_opp_wager == 0 and state.my_wager == 0:
                self.opp_raise_count += 1
                if pot_before > 0:
                    self.opp_bet_sizes.append(inc / pot_before)
            elif inc > state.my_wager - self.prev_opp_wager:
                self.opp_raise_count += 1
                if pot_before > 0:
                    self.opp_bet_sizes.append(inc / pot_before)
            else:
                self.opp_call_count += 1
            self.opp_total_actions += 1
        self.prev_opp_wager = ow

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState):
        self.hands_played += 1
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.we_have_initiative = False

        if game_info.time_bank > 5.0:
            r1, r2 = current_state.my_hand[0][0], current_state.my_hand[1][0]
            suited = current_state.my_hand[0][1] == current_state.my_hand[1][1]
            ranks = tuple(sorted([r1, r2], key=lambda x: RANK_VALUES[x], reverse=True))
            key = (ranks[0], ranks[1], suited)
            if key not in self._preflop_cache:
                n = 300 if game_info.time_bank > 12.0 else 180
                self._preflop_cache[key] = monte_carlo_equity(
                    current_state.my_hand, [], [], n
                )
            self.preflop_equity = self._preflop_cache[key]
        else:
            self.preflop_equity = fast_hand_rank(current_state.my_hand)

        self.hand_equity = self.preflop_equity

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState):
        payoff = current_state.payoff
        
        if payoff > 0 and not current_state.opp_revealed_cards:
            if self.hand_aggression > 0:
                self.opp_fold_count += 1
                self.opp_total_actions += 1

    def _pot_odds(self, state: PokerState) -> float:
        cost = state.cost_to_call
        if cost <= 0:
            return 0.0
        return cost / (state.pot + cost)

    def _geometric_bet_size(self, state, equity):
        pot = state.pot
        my_chips = state.my_chips
        opp_chips = state.opp_chips
        effective_stack = min(my_chips, opp_chips)
        spr = effective_stack / max(pot, 1)

        if equity > 0.80:
            bet_frac = 0.80
        elif equity > 0.65:
            bet_frac = 0.60
        elif equity > 0.50:
            bet_frac = 0.48
        else:
            opp_fold = self._opp_fold_rate()
            if opp_fold > 0.35:
                bet_frac = 0.32
            else:
                bet_frac = 0.40

        if state.street == "river" and equity > 0.75:
            bet_frac = min(bet_frac * 1.2, 1.0)

        target = int(pot * bet_frac)
        min_raise, max_raise = state.raise_bounds
        return max(min_raise, min(target + state.my_wager, max_raise))

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street

        if street != "auction":
            self._infer_opp(current_state)

        my_chips = current_state.my_chips
        opp_chips = current_state.opp_chips

        if street == "pre-flop":
            equity = self.preflop_equity
        elif street == "auction":
            equity = self.hand_equity
        else:
            if game_info.time_bank > 12.0:
                equity = monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    current_state.opp_revealed_cards,
                    250
                )
            elif game_info.time_bank > 5.0:
                equity = monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    current_state.opp_revealed_cards,
                    120
                )
            else:
                equity = fast_hand_rank(current_state.my_hand)
            self.hand_equity = equity

        if street == "auction":
            return self._handle_auction(current_state, equity)

        if street == "pre-flop":
            return self._handle_preflop(current_state, equity)

        return self._handle_postflop(current_state, equity)

    def _handle_auction(self, state: PokerState, equity: float) -> ActionBid:
        my_chips = state.my_chips
        pot = state.pot
        opp_avg = self._opp_avg_bid()
        opp_std = self._opp_bid_std()

        if equity > 0.80:
            info_value = 0.10
        elif equity > 0.65:
            info_value = 0.06 + (equity - 0.65) * 0.5
        elif equity > 0.50:
            info_value = 0.08 + (0.65 - abs(equity - 0.575)) * 0.35
        elif equity > 0.38:
            info_value = 0.04
        else:
            info_value = 0.01

        base_bid = int(info_value * pot)

        if len(self.opp_bids) >= 8:
            if equity > 0.55:
                adaptive_bid = int(opp_avg + opp_std * 0.3) + 1
                base_bid = max(base_bid, adaptive_bid)
            elif equity < 0.35:
                base_bid = 0
            if opp_avg < 5 and equity > 0.45:
                base_bid = max(base_bid, 2)
        elif len(self.opp_bids) >= 3:
            if equity > 0.55:
                adaptive_bid = int(opp_avg) + 2
                base_bid = max(base_bid, adaptive_bid)

        if equity > 0.70:
            cap = int(my_chips * 0.18)
        elif equity > 0.50:
            cap = int(my_chips * 0.10)
        else:
            cap = int(my_chips * 0.04)

        bid = max(0, min(base_bid, cap, my_chips))
        
        if equity > 0.40 and bid < 2 and my_chips > 100:
            bid = max(bid, 2)
            
        return ActionBid(bid)

    def _handle_preflop(self, state: PokerState, equity: float):
        cost = state.cost_to_call
        in_position = state.is_bb
        pot_odds = self._pot_odds(state)

        if cost > 0:
            if equity > 0.72:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))
                return ActionCall()

            if equity > pot_odds + 0.05:
                return ActionCall()

            if in_position and equity > pot_odds - 0.02:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()
        else:
            if state.can_act(ActionRaise):
                if equity > 0.52:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))

                if equity > 0.40 and random.random() < 0.48:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    min_raise, _ = state.raise_bounds
                    return ActionRaise(min_raise)

            return ActionCheck()

    def _handle_postflop(self, state: PokerState, equity: float):
        street = state.street
        cost = state.cost_to_call
        pot = state.pot

        pot_odds = self._pot_odds(state)
        opp_fold = self._opp_fold_rate()
        opp_agg = self._opp_aggression()

        if cost > 0:
            implied_mult = 0.78 if street in ("flop", "turn") else 1.0
            effective_odds = pot_odds * implied_mult

            if equity > 0.76:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))
                return ActionCall()

            if equity > effective_odds + 0.035:
                return ActionCall()

            if street in ("flop", "turn") and equity + 0.12 > effective_odds:
                if random.random() < 0.65:
                    return ActionCall()

            if opp_agg > 2.0 and equity > effective_odds - 0.04:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()
        else:
            if state.can_act(ActionRaise):
                cbet_ok = self.we_have_initiative and street == "flop" and equity > 0.40
                value_bet = equity > 0.58
                thin_bet = equity > 0.48 and random.random() < 0.45

                if cbet_ok or value_bet or thin_bet:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))

                if equity < 0.30 and opp_fold > 0.30 and self.hand_aggression < 2:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))

                if not self.we_have_initiative and street in ("turn", "river") and equity > 0.54 and random.random() < 0.32:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._geometric_bet_size(state, equity))

        return ActionCheck()


if __name__ == "__main__":
    run_bot(ICMPlayerV3(), parse_args())
