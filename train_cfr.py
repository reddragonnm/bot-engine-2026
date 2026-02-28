"""
Offline Monte Carlo CFR trainer for Sneak Peek Hold'em.

Produces strategy tables via self-play that converge toward Nash equilibrium.
Saves the average strategy to a JSON file that the competition bot loads.

Usage:
    python train_cfr.py [--iterations N] [--output FILE]

Architecture:
    - External Sampling MCCFR (samples opponent actions + chance)
    - Information sets keyed by: (street, hand_bucket, board_bucket, pot_bucket,
      cost_bucket, position, auction_info, history_summary)
    - Abstract actions: fold, check, call, raise_small, raise_medium, raise_large,
      bid_0, bid_low, bid_medium, bid_high
    - Hand strength computed via eval7 Monte Carlo equity (fast, cached)
    - Strategy stored as cumulative regret + cumulative strategy tables
"""

import eval7
import random
import math
import json
import time
import argparse
import os

# =============================================================================
# Game Constants
# =============================================================================
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
A_RAISE_S = 3   # ~40% pot
A_RAISE_M = 4   # ~70% pot
A_RAISE_L = 5   # all-in or ~pot
A_BID_0 = 6
A_BID_L = 7
A_BID_M = 8
A_BID_H = 9

ACTION_NAMES = [
    "fold", "check", "call", "raise_s", "raise_m", "raise_l",
    "bid_0", "bid_l", "bid_m", "bid_h",
]

NUM_ACTIONS = 10

# =============================================================================
# Hand Strength Bucketing
# =============================================================================

def fast_hand_rank(cards):
    """Quick preflop hand strength estimate, returns 0-1."""
    if len(cards) != 2:
        return 0.5
    r1 = RANK_VALUES.get(str(cards[0])[0], 0)
    r2 = RANK_VALUES.get(str(cards[1])[0], 0)
    suited = str(cards[0])[1] == str(cards[1])[1]
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


def monte_carlo_equity(my_cards, board_cards, opp_known, remaining, n_sims=100):
    """Estimate equity via Monte Carlo. All args are eval7.Card lists."""
    wins = ties = total = 0
    board_need = 5 - len(board_cards)
    opp_need = 2 - len(opp_known)

    for _ in range(n_sims):
        random.shuffle(remaining)
        idx = 0
        oh = list(opp_known)
        for _ in range(opp_need):
            oh.append(remaining[idx])
            idx += 1
        sb = list(board_cards)
        for _ in range(board_need):
            sb.append(remaining[idx])
            idx += 1
        ms = eval7.evaluate(sb + list(my_cards))
        os_ = eval7.evaluate(sb + oh)
        if ms > os_:
            wins += 1
        elif ms == os_:
            ties += 1
        total += 1
    return (wins + ties * 0.5) / total if total else 0.5


def equity_bucket(eq, n_buckets=8):
    """Map equity [0,1] to bucket [0, n_buckets-1]."""
    return min(int(eq * n_buckets), n_buckets - 1)


def pot_bucket(pot):
    """Discretize pot size."""
    r = pot / BIG_BLIND
    if r < 3:
        return 0
    if r < 8:
        return 1
    if r < 20:
        return 2
    if r < 50:
        return 3
    return 4


def cost_bucket(cost, pot):
    """Discretize cost-to-call relative to pot."""
    if cost <= 0:
        return 0
    if pot <= 0:
        return 3
    ratio = cost / pot
    if ratio < 0.25:
        return 1
    if ratio < 0.60:
        return 2
    return 3


# =============================================================================
# Simplified Game State for Training
# =============================================================================

class TrainState:
    """Minimal game state for CFR traversal."""
    __slots__ = [
        'street', 'auction', 'dealer',
        'hands', 'board', 'deck_remaining',
        'wagers', 'chips', 'bids',
        'opp_revealed', 'history',
    ]

    def __init__(self):
        self.street = 0        # 0=preflop, 3=flop, 4=turn, 5=river
        self.auction = False
        self.dealer = 0        # action counter, active = dealer % 2
        self.hands = [[], []]  # eval7.Card lists
        self.board = []        # eval7.Card list
        self.deck_remaining = []
        self.wagers = [SMALL_BLIND, BIG_BLIND]
        self.chips = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        self.bids = [None, None]
        self.opp_revealed = [[], []]  # cards revealed to each player
        self.history = []  # list of (street, action_id) for info set

    def copy(self):
        s = TrainState()
        s.street = self.street
        s.auction = self.auction
        s.dealer = self.dealer
        s.hands = [list(h) for h in self.hands]
        s.board = list(self.board)
        s.deck_remaining = list(self.deck_remaining)
        s.wagers = list(self.wagers)
        s.chips = list(self.chips)
        s.bids = list(self.bids)
        s.opp_revealed = [list(r) for r in self.opp_revealed]
        s.history = list(self.history)
        return s

    @property
    def active(self):
        return self.dealer % 2

    @property
    def pot(self):
        return (STARTING_STACK - self.chips[0]) + (STARTING_STACK - self.chips[1])

    @property
    def cost_to_call(self):
        a = self.active
        return max(0, self.wagers[1 - a] - self.wagers[a])

    def street_name(self):
        if self.auction:
            return "auction"
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(self.street, "unknown")

    def legal_abstract_actions(self):
        """Return list of abstract action IDs that are legal."""
        if self.auction:
            return [A_BID_0, A_BID_L, A_BID_M, A_BID_H]

        a = self.active
        cost = self.cost_to_call
        can_raise = self.chips[a] > 0 and self.chips[1 - a] > 0

        if cost == 0:
            # Check or raise
            if not can_raise or (self.chips[a] == 0):
                return [A_CHECK]
            return [A_CHECK, A_RAISE_S, A_RAISE_M, A_RAISE_L]
        else:
            # Fold, call, or raise
            # Can't re-raise if cost uses all our chips or opp is all-in
            if cost >= self.chips[a] or self.chips[1 - a] == 0:
                return [A_FOLD, A_CALL]
            if not can_raise:
                return [A_FOLD, A_CALL]
            return [A_FOLD, A_CALL, A_RAISE_S, A_RAISE_M, A_RAISE_L]

    def compute_raise_amount(self, action_id):
        """Convert abstract raise to concrete amount (total wager)."""
        a = self.active
        opp = 1 - a
        cost = self.cost_to_call
        pot = self.pot

        max_bet = min(self.chips[a], self.chips[opp] + cost)
        min_bet = min(max_bet, cost + max(cost, BIG_BLIND))

        if action_id == A_RAISE_S:
            target = int(pot * 0.40)
        elif action_id == A_RAISE_M:
            target = int(pot * 0.70)
        else:  # A_RAISE_L
            target = max_bet  # all-in

        bet = max(min_bet, min(target, max_bet))
        return self.wagers[a] + bet

    def compute_bid_amount(self, action_id):
        """Convert abstract bid to concrete amount."""
        a = self.active
        chips = self.chips[a]
        pot = self.pot

        if action_id == A_BID_0:
            return 0
        elif action_id == A_BID_L:
            return max(1, min(int(pot * 0.03), chips))
        elif action_id == A_BID_M:
            return max(2, min(int(pot * 0.10), chips))
        else:  # A_BID_H
            return max(5, min(int(pot * 0.20), chips))

    def apply_action(self, action_id):
        """
        Apply an abstract action. Returns (new_state, terminal, payoffs).
        payoffs is [p0_delta, p1_delta] if terminal, else None.
        """
        s = self.copy()
        a = s.active
        opp = 1 - a
        s.history.append((s.street_name(), action_id))

        # --- Auction ---
        if s.auction:
            bid = s.compute_bid_amount(action_id)
            s.bids[a] = bid
            if None not in s.bids:
                # Resolve auction
                if s.bids[0] == s.bids[1]:
                    # Tie: both pay own bid, both see a card
                    s.chips[0] -= s.bids[0]
                    s.chips[1] -= s.bids[1]
                    if s.hands[1]:
                        s.opp_revealed[0].append(random.choice(s.hands[1]))
                    if s.hands[0]:
                        s.opp_revealed[1].append(random.choice(s.hands[0]))
                else:
                    winner = 0 if s.bids[0] > s.bids[1] else 1
                    loser = 1 - winner
                    s.chips[winner] -= s.bids[loser]  # winner pays loser's bid
                    if s.hands[loser]:
                        s.opp_revealed[winner].append(random.choice(s.hands[loser]))
                s.auction = False
                s.wagers = [0, 0]
                s.dealer = 1  # BB acts first post-flop
            else:
                s.dealer += 1
            return s, False, None

        # --- Fold ---
        if action_id == A_FOLD:
            if a == 0:
                delta = s.chips[0] - STARTING_STACK
            else:
                delta = STARTING_STACK - s.chips[1]
            return s, True, [delta, -delta]

        # --- Check ---
        if action_id == A_CHECK:
            # Check if street is over
            street_over = False
            if s.street == 0 and s.dealer > 0:
                street_over = True
            elif s.dealer > 1:
                street_over = True

            if street_over:
                return s._next_street()
            else:
                s.dealer += 1
                return s, False, None

        # --- Call ---
        if action_id == A_CALL:
            cost = s.cost_to_call
            if s.street == 0 and s.dealer == 0:
                # SB completing to BB
                added = BIG_BLIND - s.wagers[a]
                s.chips[a] -= added
                s.wagers[a] = BIG_BLIND
            else:
                s.chips[a] -= cost
                s.wagers[a] = s.wagers[opp]
            return s._next_street()

        # --- Raise ---
        if action_id in (A_RAISE_S, A_RAISE_M, A_RAISE_L):
            amount = s.compute_raise_amount(action_id)
            added = amount - s.wagers[a]
            s.chips[a] -= added
            s.wagers[a] = amount
            s.dealer += 1
            return s, False, None

        # Should not reach here
        return s, False, None

    def _next_street(self):
        """Advance to next street or showdown."""
        if self.street == 5:
            # Showdown
            return self._showdown()

        if self.street == 0:
            # Preflop -> Auction -> Flop
            self.street = 3
            self.auction = True
            self.bids = [None, None]
            self.wagers = [0, 0]
            self.dealer = 0
            # Deal flop
            self.board = list(self.deck_remaining[:3])
        elif self.street == 3:
            self.street = 4
            self.wagers = [0, 0]
            self.dealer = 1
            self.board = list(self.deck_remaining[:4])
        elif self.street == 4:
            self.street = 5
            self.wagers = [0, 0]
            self.dealer = 1
            self.board = list(self.deck_remaining[:5])

        return self, False, None

    def _showdown(self):
        """Evaluate hands and return payoffs."""
        board5 = list(self.deck_remaining[:5])
        score0 = eval7.evaluate(board5 + list(self.hands[0]))
        score1 = eval7.evaluate(board5 + list(self.hands[1]))
        if score0 > score1:
            delta = STARTING_STACK - self.chips[1]
        elif score0 < score1:
            delta = self.chips[0] - STARTING_STACK
        else:
            delta = (self.chips[0] - self.chips[1]) // 2
        return self, True, [delta, -delta]

    def get_info_set_key(self, player):
        """
        Information set key for the given player.
        Contains everything this player can observe.
        """
        # Hand strength bucket
        my_hand = self.hands[player]
        opp_known = self.opp_revealed[player]
        board = self.board if not self.auction else []

        known = set(my_hand + board + opp_known)
        # Build remaining deck excluding known cards (but also excluding
        # opponent's actual hand since we don't know it)
        remaining = [c for c in self.deck_remaining if c not in known]

        if self.street == 0 or self.auction:
            eq = fast_hand_rank([str(c) for c in my_hand])
        else:
            if len(remaining) >= (5 - len(board)) + (2 - len(opp_known)):
                eq = monte_carlo_equity(my_hand, board, opp_known, remaining, n_sims=50)
            else:
                eq = fast_hand_rank([str(c) for c in my_hand])

        eb = equity_bucket(eq)
        pb = pot_bucket(self.pot)
        cb = cost_bucket(self.cost_to_call, self.pot)
        pos = player  # 0=SB, 1=BB
        sn = self.street_name()

        # Has opp info from auction?
        has_info = 1 if opp_known else 0

        # Compact action history for this street
        street_hist = []
        for sh, ah in self.history:
            if sh == sn:
                street_hist.append(ah)
        # Limit to last 3 actions to keep info sets manageable
        hist_key = tuple(street_hist[-3:])

        return (sn, eb, pb, cb, pos, has_info, hist_key)


def deal_hand():
    """Deal a new hand and return initial TrainState."""
    deck = eval7.Deck()
    deck.shuffle()
    all_cards = deck.deal(52)

    s = TrainState()
    s.hands = [list(all_cards[0:2]), list(all_cards[2:4])]
    # deck_remaining[0:5] will be the board cards, rest are for MC sampling
    s.deck_remaining = list(all_cards[4:])
    s.board = []
    return s


# =============================================================================
# CFR Trainer
# =============================================================================

class CFRTrainer:
    """External Sampling Monte Carlo CFR."""

    def __init__(self):
        # regret_sum[info_set_key] = {action_id: float}
        self.regret_sum = {}
        # strategy_sum[info_set_key] = {action_id: float}
        self.strategy_sum = {}
        self.iterations = 0

    def get_strategy(self, info_set, legal_actions):
        """
        Compute current strategy from regret-matching.
        Returns dict {action_id: probability}.
        """
        key = info_set
        regrets = self.regret_sum.get(key, {})

        # Regret matching
        positive_regrets = {}
        total = 0.0
        for a in legal_actions:
            r = max(0.0, regrets.get(a, 0.0))
            positive_regrets[a] = r
            total += r

        strategy = {}
        if total > 0:
            for a in legal_actions:
                strategy[a] = positive_regrets[a] / total
        else:
            # Uniform
            p = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[a] = p

        return strategy

    def update_strategy_sum(self, info_set, strategy, reach_prob):
        """Accumulate strategy for averaging."""
        if info_set not in self.strategy_sum:
            self.strategy_sum[info_set] = {}
        ss = self.strategy_sum[info_set]
        for a, p in strategy.items():
            ss[a] = ss.get(a, 0.0) + reach_prob * p

    def get_average_strategy(self, info_set, legal_actions):
        """Get the average strategy (converges to Nash)."""
        ss = self.strategy_sum.get(info_set, {})
        total = sum(ss.get(a, 0.0) for a in legal_actions)
        if total > 0:
            return {a: ss.get(a, 0.0) / total for a in legal_actions}
        p = 1.0 / len(legal_actions)
        return {a: p for a in legal_actions}

    def cfr(self, state, traverser, reach_probs):
        """
        External sampling MCCFR traversal.
        traverser: which player we're computing regrets for (0 or 1).
        reach_probs: [p0_reach, p1_reach]
        Returns expected value for traverser.
        """
        active = state.active

        # Terminal check
        legal = state.legal_abstract_actions()

        # Get info set
        info_set = state.get_info_set_key(active)
        strategy = self.get_strategy(info_set, legal)

        if active == traverser:
            # Traverser's node: compute regrets for all actions
            action_values = {}
            node_value = 0.0

            for a in legal:
                new_state, terminal, payoffs = state.apply_action(a)
                if terminal:
                    action_values[a] = payoffs[traverser]
                else:
                    action_values[a] = self.cfr(new_state, traverser, reach_probs)
                node_value += strategy[a] * action_values[a]

            # Update regrets
            if info_set not in self.regret_sum:
                self.regret_sum[info_set] = {}
            rs = self.regret_sum[info_set]
            for a in legal:
                regret = action_values[a] - node_value
                rs[a] = rs.get(a, 0.0) + regret

            # Update strategy sum
            self.update_strategy_sum(info_set, strategy, reach_probs[traverser])

            return node_value
        else:
            # Opponent's node: sample one action according to strategy
            r = random.random()
            cum = 0.0
            chosen = legal[-1]
            for a in legal:
                cum += strategy.get(a, 0.0)
                if r < cum:
                    chosen = a
                    break

            new_reach = list(reach_probs)
            new_reach[active] *= strategy[chosen]

            new_state, terminal, payoffs = state.apply_action(chosen)
            if terminal:
                return payoffs[traverser]
            return self.cfr(new_state, traverser, new_reach)

    def train_iteration(self):
        """Run one iteration of MCCFR (one hand, both players)."""
        state = deal_hand()

        # Traverse for player 0
        self.cfr(state.copy(), 0, [1.0, 1.0])
        # Traverse for player 1
        self.cfr(state.copy(), 1, [1.0, 1.0])

        self.iterations += 1

    def export_strategy(self):
        """
        Export the average strategy as a JSON-serializable dict.
        Format: {info_set_str: {action_name: probability}}
        """
        result = {}
        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total <= 0:
                continue

            # Convert info set tuple to string key
            key = str(info_set)
            probs = {}
            for a, s in action_sums.items():
                p = s / total
                if p > 0.001:  # prune tiny probabilities
                    probs[str(a)] = round(p, 4)

            if probs:
                result[key] = probs

        return result

    def apply_regret_discount(self, discount=0.995):
        """Apply discounting to regrets to speed convergence (DCFR)."""
        for info_set in self.regret_sum:
            rs = self.regret_sum[info_set]
            for a in rs:
                rs[a] *= discount


def main():
    parser = argparse.ArgumentParser(description="Train CFR poker strategy")
    parser.add_argument("--iterations", type=int, default=100000,
                        help="Number of MCCFR iterations")
    parser.add_argument("--output", type=str, default="cfr_strategy.json",
                        help="Output strategy file")
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")
    args = parser.parse_args()

    trainer = CFRTrainer()

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        with open(args.resume, "r") as f:
            checkpoint = json.load(f)
        # Reconstruct regret_sum and strategy_sum
        for key_str, regrets in checkpoint.get("regret_sum", {}).items():
            key = eval(key_str)
            trainer.regret_sum[key] = {int(a): v for a, v in regrets.items()}
        for key_str, strats in checkpoint.get("strategy_sum", {}).items():
            key = eval(key_str)
            trainer.strategy_sum[key] = {int(a): v for a, v in strats.items()}
        trainer.iterations = checkpoint.get("iterations", 0)
        print(f"Resumed at iteration {trainer.iterations}")

    print(f"Training MCCFR for {args.iterations} iterations...")
    print(f"Output: {args.output}")

    start_time = time.time()
    target = trainer.iterations + args.iterations

    while trainer.iterations < target:
        trainer.train_iteration()

        # Discount regrets periodically (DCFR)
        if trainer.iterations % 1000 == 0:
            trainer.apply_regret_discount(0.998)

        if trainer.iterations % args.checkpoint_interval == 0:
            elapsed = time.time() - start_time
            n_infosets = len(trainer.regret_sum)
            its_per_sec = args.checkpoint_interval / elapsed if elapsed > 0 else 0
            print(
                f"  Iteration {trainer.iterations:>8d} | "
                f"Info sets: {n_infosets:>6d} | "
                f"Speed: {its_per_sec:.0f} it/s | "
                f"Elapsed: {elapsed:.1f}s"
            )
            start_time = time.time()

            # Save checkpoint
            checkpoint = {
                "iterations": trainer.iterations,
                "regret_sum": {
                    str(k): {str(a): v for a, v in rs.items()}
                    for k, rs in trainer.regret_sum.items()
                },
                "strategy_sum": {
                    str(k): {str(a): v for a, v in ss.items()}
                    for k, ss in trainer.strategy_sum.items()
                },
            }
            ckpt_file = args.output.replace(".json", "_checkpoint.json")
            with open(ckpt_file, "w") as f:
                json.dump(checkpoint, f)

    # Export final strategy
    print(f"\nTraining complete. {trainer.iterations} total iterations.")
    print(f"Total info sets: {len(trainer.regret_sum)}")

    strategy = trainer.export_strategy()
    print(f"Exported strategy entries: {len(strategy)}")

    with open(args.output, "w") as f:
        json.dump(strategy, f)
    print(f"Strategy saved to {args.output}")

    # Also save full checkpoint for resuming
    checkpoint = {
        "iterations": trainer.iterations,
        "regret_sum": {
            str(k): {str(a): v for a, v in rs.items()}
            for k, rs in trainer.regret_sum.items()
        },
        "strategy_sum": {
            str(k): {str(a): v for a, v in ss.items()}
            for k, ss in trainer.strategy_sum.items()
        },
    }
    ckpt_file = args.output.replace(".json", "_checkpoint.json")
    with open(ckpt_file, "w") as f:
        json.dump(checkpoint, f)
    print(f"Checkpoint saved to {ckpt_file}")


if __name__ == "__main__":
    main()
