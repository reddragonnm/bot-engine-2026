'''
All-In Pokerbot, written in Python.
This bot always bids its entire stack and raises to the maximum possible amount.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

class Player(BaseBot):
    '''
    An aggressive pokerbot that always goes All-In.
    '''

    def __init__(self) -> None:
        pass

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        pass

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        '''
        Always selects the most aggressive action available.
        '''

        # 1. Auction: Bid the entire remaining stack
        if current_state.street == 'auction':
            # In a second-price auction, bidding your full stack maximizes your 
            # chance to win while only paying the opponent's bid.
            return ActionBid(current_state.my_chips)

        # 2. Betting: Always Raise to the maximum allowed amount
        if current_state.can_act(ActionRaise):
            # The maximum legal raise is bounded by the players' stack sizes.
            min_raise, max_raise = current_state.raise_bounds
            return ActionRaise(max_raise)

        # 3. If we cannot raise (e.g., already All-In or opponent is All-In), we Call
        if current_state.can_act(ActionCall):
            return ActionCall()
        
        # 4. Fallback to Check if no other action is required
        return ActionCheck()

if __name__ == '__main__':
    run_bot(Player(), parse_args())