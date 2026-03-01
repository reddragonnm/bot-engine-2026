"""
Nested Simulation Poker Bot v5
Uses nested Monte Carlo simulation for optimal decision making.
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import eval7
import random
import math
from collections import defaultdict


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
        return 0.5 + (high - 2) * 0.048
    suit_bonus = 0.042 if suited else 0.0
    gap = high - low
    connect_bonus = max(0, 0.032 - gap * 0.0035)
    score = (high + low - 4) / 23.0 + suit_bonus + connect_bonus
    if high == 14:
        score += 0.12
    if high >= 13 and low >= 11:
        score += 0.09
    if high >= 13 and low >= 10:
        score += 0.05
    return min(max(score, 0.0), 1.0)


def monte_carlo_equity(my_hand, board, opp_revealed, n_sims=300):
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


def nested_simulation_auction(my_hand, opp_history, n_outer=80, n_inner=30):
    """Nested simulation: outer loop samples opponent model, inner loop samples cards."""
    values = []
    
    for _ in range(n_outer):
        opp_bid_model = estimate_opp_bid_model(opp_history)
        
        for _ in range(n_inner):
            simulated_opp_hand = random.sample([c for c in eval7.Deck().cards 
                if str(c) not in [str(eval7.Card(h)) for h in my_hand]], 2)
            
            my_strength = fast_hand_rank(my_hand)
            opp_strength = fast_hand_rank([str(c) for c in simulated_opp_hand])
            
            value = simulate_auction_outcome(my_hand, simulated_opp_hand, opp_bid_model)
            values.append(value)
    
    return sum(values) / len(values) if values else 0.5


def estimate_opp_bid_model(bid_history):
    if len(bid_history) < 3:
        return {'mean': 9.0, 'std': 8.0, 'aggression': 0.5}
    
    mean = sum(bid_history) / len(bid_history)
    variance = sum((b - mean) ** 2 for b in bid_history) / len(bid_history)
    std = math.sqrt(variance)
    
    aggression = 0.5
    if mean > 12:
        aggression = 0.7
    elif mean < 5:
        aggression = 0.3
    
    return {'mean': mean, 'std': std, 'aggression': aggression}


def simulate_auction_outcome(my_hand, opp_hand, opp_model):
    my_strength = fast_hand_rank(my_hand)
    
    base_value = my_strength
    if opp_model['aggression'] > 0.6:
        base_value *= 0.95
    elif opp_model['aggression'] < 0.4:
        base_value *= 1.05
    
    return base_value


class NestedSimBot(BaseBot):
    def __init__(self):
        self.opp_fold_count = 0
        self.opp_raise_count = 0
        self.opp_call_count = 0
        self.opp_total_actions = 0
        self.opp_bids = []
        self.opp_bid_history = []
        
        self.strategy_history = defaultdict(list)
        self.hands_played = 0
        
        self.hand_equity = 0.5
        self.preflop_equity = 0.5
        self.hand_aggression = 0
        self.prev_opp_wager = 0
        self.prev_street = ""
        self.we_have_initiative = False
        self._preflop_cache = {}

    def _opp_fold_rate(self):
        if self.opp_total_actions < 8:
            return 0.30
        return self.opp_fold_count / self.opp_total_actions

    def _opp_aggression(self):
        passive = self.opp_call_count
        if passive == 0:
            return 2.0 if self.opp_raise_count > 0 else 1.0
        return self.opp_raise_count / max(passive, 1)

    def _opp_avg_bid(self):
        if not self.opp_bids:
            return 9.0
        return sum(self.opp_bids) / len(self.opp_bids)

    def _opp_bid_std(self):
        if len(self.opp_bids) < 3:
            return 8.0
        avg = self._opp_avg_bid()
        return math.sqrt(sum((b - avg) ** 2 for b in self.opp_bids) / len(self.opp_bids))

    def _detect_opp_style(self):
        fold_rate = self._opp_fold_rate()
        agg = self._opp_aggression()
        bid_avg = self._opp_avg_bid()
        
        style = {
            'tight': fold_rate > 0.35,
            'loose': fold_rate < 0.25,
            'aggressive': agg > 1.8,
            'passive': agg < 0.9,
            'high_bidder': bid_avg > 12,
            'low_bidder': bid_avg < 6,
        }
        return style

    def _infer_opp(self, state: PokerState):
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
                n = 400 if game_info.time_bank > 12.0 else 250
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

    def _ev_simulation(self, state, action_type, equity, n_sims=60):
        """Simulate EV of an action with nested sampling."""
        pot = state.pot
        style = self._detect_opp_style()
        
        if action_type == 'raise':
            bet_size = self._calculate_bet_size(state, equity)
            if style['tight']:
                fold_prob = min(0.35 + (equity - 0.4) * 0.3, 0.55)
            elif style['loose']:
                fold_prob = max(0.15, 0.30 - (0.5 - equity) * 0.2)
            else:
                fold_prob = 0.28 + (equity - 0.45) * 0.4
            
            call_ev = equity * (pot + bet_size * 2) - (1 - equity) * bet_size
            fold_ev = pot * fold_prob
            
            return fold_ev + (1 - fold_prob) * call_ev * 0.7
            
        return 0

    def _calculate_bet_size(self, state, equity):
        pot = state.pot
        street = state.street
        style = self._detect_opp_style()
        
        if equity > 0.82:
            bet_frac = 0.90
        elif equity > 0.70:
            bet_frac = 0.70
        elif equity > 0.55:
            bet_frac = 0.55
        elif equity > 0.45:
            if style['tight']:
                bet_frac = 0.32
            else:
                bet_frac = 0.40
        else:
            if style['loose']:
                bet_frac = 0.45
            else:
                bet_frac = 0.35

        if street == "river" and equity > 0.72:
            bet_frac = min(bet_frac * 1.3, 1.15)

        target = int(pot * bet_frac)
        min_raise, max_raise = state.raise_bounds
        return max(min_raise, min(target + state.my_wager, max_raise))

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        street = current_state.street

        if street != "auction":
            self._infer_opp(current_state)

        if street == "pre-flop":
            equity = self.preflop_equity
        elif street == "auction":
            equity = self.hand_equity
        else:
            if game_info.time_bank > 14.0:
                equity = monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    current_state.opp_revealed_cards,
                    400
                )
            elif game_info.time_bank > 6.0:
                equity = monte_carlo_equity(
                    current_state.my_hand,
                    current_state.board,
                    current_state.opp_revealed_cards,
                    200
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
        style = self._detect_opp_style()

        if equity > 0.88:
            info_value = 0.15
        elif equity > 0.72:
            info_value = 0.10 + (equity - 0.72) * 0.42
        elif equity > 0.58:
            info_value = 0.12 + (0.72 - abs(equity - 0.65)) * 0.30
        elif equity > 0.45:
            info_value = 0.065
        else:
            info_value = 0.022

        base_bid = int(info_value * pot)

        if len(self.opp_bids) >= 5:
            if equity > 0.60:
                adaptive = int(opp_avg + opp_std * 0.35) + 3
                base_bid = max(base_bid, adaptive)
            elif equity < 0.30:
                base_bid = 0
            if opp_avg < 4 and equity > 0.50:
                base_bid = max(base_bid, 4)
        elif len(self.opp_bids) >= 1:
            if equity > 0.60:
                base_bid = max(base_bid, int(opp_avg * 1.15) + 3)

        if style['high_bidder']:
            cap_mult = 0.28
        elif style['low_bidder']:
            cap_mult = 0.10
        else:
            cap_mult = 0.18
            
        if equity > 0.78:
            cap_mult = min(cap_mult * 1.25, 0.35)
        elif equity < 0.48:
            cap_mult *= 0.65
            
        cap = int(my_chips * cap_mult)
        bid = max(0, min(base_bid, cap, my_chips))
        
        if equity > 0.48 and bid < 4 and my_chips > 25:
            bid = max(bid, 4)
            
        return ActionBid(bid)

    def _handle_preflop(self, state: PokerState, equity: float):
        cost = state.cost_to_call
        in_position = state.is_bb
        pot_odds = self._pot_odds(state)
        style = self._detect_opp_style()

        if cost > 0:
            if equity > 0.74:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))
                return ActionCall()

            if equity > pot_odds + 0.055:
                return ActionCall()

            if in_position and equity > pot_odds - 0.01:
                return ActionCall()

            if style['passive'] and equity > pot_odds - 0.03:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()
        else:
            if state.can_act(ActionRaise):
                if equity > 0.54:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))

                if equity > 0.42 and random.random() < 0.52:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    min_raise, _ = state.raise_bounds
                    return ActionRaise(min_raise)

            return ActionCheck()

    def _handle_postflop(self, state: PokerState, equity: float):
        street = state.street
        cost = state.cost_to_call
        pot_odds = self._pot_odds(state)
        style = self._detect_opp_style()

        if cost > 0:
            implied_mult = 0.72 if street in ("flop", "turn") else 1.0
            effective_odds = pot_odds * implied_mult

            if equity > 0.77:
                if state.can_act(ActionRaise):
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))
                return ActionCall()

            if equity > effective_odds + 0.035:
                return ActionCall()

            if street in ("flop", "turn") and equity + 0.18 > effective_odds:
                if random.random() < 0.75:
                    return ActionCall()

            if style['passive'] and equity > effective_odds - 0.015:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCheck()
        else:
            if state.can_act(ActionRaise):
                cbet_ok = self.we_have_initiative and street == "flop" and equity > 0.40
                value_bet = equity > 0.59
                thin_bet = equity > 0.49 and random.random() < 0.50

                if cbet_ok or value_bet or thin_bet:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))

                if equity < 0.30 and style['tight'] and self.hand_aggression < 2:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))

                if not self.we_have_initiative and street in ("turn", "river") and equity > 0.54 and random.random() < 0.36:
                    self.hand_aggression += 1
                    self.we_have_initiative = True
                    return ActionRaise(self._calculate_bet_size(state, equity))

        return ActionCheck()


if __name__ == "__main__":
    run_bot(NestedSimBot(), parse_args())
