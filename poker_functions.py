# poker_treys_equity.py
#
# Use Treys for Monte-Carlo equity and hand-type flags at any stage,
# with safe evaluation to avoid KeyError bugs in Treys.

import random
from treys import Card, Deck, Evaluator
from typing import List, Dict, Tuple
from itertools import combinations


def parse_cards(cards_str: str) -> List[int]:
    """
    Convert a string like "Jc9h5c" or "Ac2d" into a list of Treys card ints.
    """
    return [Card.new(cards_str[i:i+2]) for i in range(0, len(cards_str), 2)]


def parse_game_log(lines: List[str]) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Parse action logs. Returns:
      - board: list of community card ints
      - hole_cards: mapping player_id -> hole card ints
    """
    board: List[int] = []
    hole_cards: Dict[str, List[int]] = {}

    for line in lines:
        parts = line.strip().split()
        if parts[:2] == ["d", "dh"] and len(parts) >= 4:
            pid, cs = parts[2], parts[3]
            if "?" not in cs:
                hole_cards[pid] = parse_cards(cs)
        elif parts[:2] == ["d", "db"] and len(parts) >= 3:
            board.extend(parse_cards(parts[2]))
        elif len(parts) >= 3 and parts[1] == "sm":
            pid, cs = parts[0], parts[2]
            hole_cards[pid] = parse_cards(cs)
    return board, hole_cards


def safe_evaluate(cards: List[int], evaluator: Evaluator) -> int:
    """
    Safely evaluate any 5- to 7-card hand by brute-forcing 5-card subsets,
    skipping invalid combos. Returns the best (lowest) score.
    """
    if len(cards) == 5:
        try:
            return evaluator._five(cards)
        except KeyError:
            return 9999

    best = 9999
    for combo in combinations(cards, 5):
        try:
            score = evaluator._five(list(combo))
        except KeyError:
            continue
        if score < best:
            best = score
    return best


def simulate_equity(hole: List[int], board: List[int], trials: int = 10000) -> float:
    """
    Monte-Carlo win-rate vs one random opponent: returns float in [0.0, 1.0].
    Ties count as half-wins. Completes board to 5 cards if needed.
    Uses safe_evaluate to avoid bugs.
    """
    wins = ties = 0
    evaluator = Evaluator()

    for _ in range(trials):
        deck = Deck()
        for c in hole + board:
            deck.cards.remove(c)
        deck.shuffle()
        opp = deck.draw(2)
        needed = max(0, 5 - len(board))
        new_board = board + deck.draw(needed)

        h_val = safe_evaluate(new_board + hole, evaluator)
        o_val = safe_evaluate(new_board + opp, evaluator)
        if h_val < o_val:
            wins += 1
        elif h_val == o_val:
            ties += 1

    return (wins + ties * 0.5) / trials


def preflop_analysis(hole: List[int], trials: int = 10000) -> Tuple[float, float]:
    """
    Monte-Carlo simulation when only hole cards are known.
    Returns (avg raw score, equity) over random boards and one opponent.
    Uses safe_evaluate to avoid errors.
    """
    total_score = wins = ties = runs = 0
    evaluator = Evaluator()

    for _ in range(trials):
        deck = Deck()
        for c in hole:
            deck.cards.remove(c)
        deck.shuffle()
        opp = deck.draw(2)
        board_sim = deck.draw(5)

        score = safe_evaluate(board_sim + hole, evaluator)
        opp_score = safe_evaluate(board_sim + opp, evaluator)
        total_score += score
        runs += 1
        if score < opp_score:
            wins += 1
        elif score == opp_score:
            ties += 1

    avg_score = total_score / runs
    equity = (wins + ties * 0.5) / runs
    return avg_score, equity


# Example usage
if __name__ == "__main__":
    log = [
        "d dh p1 8sQc",
        "d dh p2 2s8d",
        "d dh p3 7dTs",
        "d dh p4 5d8h",
        "d dh p5 2h9s",
        "d dh p6 6cQd",
        "p3 f",
        "p4 f",
        "p5 f",
        "p6 f",
        "p1 cbr 300",
        "p2 f"
    ]
    board, holes = parse_game_log(log)
    hole1 = holes.get("p1", [])

    evaluator = Evaluator()
    hand_types = [
        ht.lower().replace(' ', '-')
        for ht in [
            "High Card", "One Pair", "Two Pair", "Trips",
            "Straight", "Flush", "Full House",
            "Quads", "Straight Flush"
        ]
    ]

    if not board:
        score, equity = preflop_analysis(hole1, trials=20000)
        print(f"P1 Avg raw score: {score:.1f}")
        print(f"P1 Equity:         {equity:.3f}")
        type_flags = {ht: False for ht in hand_types}
        print(f"P1 Hand types:     {type_flags}")
    else:
        equity = simulate_equity(hole1, board, trials=20000)
        print(f"P1 Equity:         {equity:.3f}")
        # Determine hand type flags
        score = safe_evaluate(board + hole1, evaluator)
        rank_class = evaluator.get_rank_class(score)
        hand_name = evaluator.class_to_string(rank_class).lower().replace(' ', '-')
        type_flags = {ht: (ht == hand_name) for ht in hand_types}
        print(f"P1 Hand types:     {type_flags}")
