# Use Treys for Monte-Carlo equity and hand-type flags at any stage,

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


def _equity_worker(args: Tuple[List[int], List[int], int]) -> Tuple[int, int, int]:
    """
    Worker for parallel equity simulation: returns (wins, ties, runs).
    """
    hole, board, trials = args
    wins = ties = runs = 0
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
        runs += 1
        if h_val < o_val:
            wins += 1
        elif h_val == o_val:
            ties += 1
    return wins, ties, runs

def simulate_equity(hole: List[int], board: List[int], trials: int = 10000, use_mp: bool = True) -> float:
    """
    Monte-Carlo win-rate vs one random opponent: returns float in [0.0, 1.0].
    If use_mp, splits work across CPU cores.
    """
    if use_mp:
        import multiprocessing
        cpu = multiprocessing.cpu_count()
        base = trials // cpu
        extras = trials % cpu
        jobs = []
        for i in range(cpu):
            n = base + (1 if i < extras else 0)
            jobs.append((hole, board, n))
        with multiprocessing.Pool(cpu) as pool:
            results = pool.map(_equity_worker, jobs)
        total_wins = total_ties = total_runs = 0
        for w, t, r in results:
            total_wins += w
            total_ties += t
            total_runs += r
        if total_runs == 0:
            return 0.0
        return (total_wins + total_ties * 0.5) / total_runs
    # fallback to single-process
    wins = ties = runs = 0
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
        runs += 1
        if h_val < o_val:
            wins += 1
        elif h_val == o_val:
            ties += 1
    if runs == 0:
        return 0.0
    return (wins + ties * 0.5) / runs


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
        "d dh p1 As6c",
      "d dh p2 9c4c",
      "d dh p3 3c5h",
      "d dh p4 6h2s",
      "d dh p5 Ks7c",
      "d dh p6 3s9s",
      "p3 f",
      "p4 f",
      "p5 f",
      "p6 f",
      "p1 cc",
      "p2 cc",
      "d db 5c8sTs",
      "p1 cc",
      "p2 cbr 150",
      "p1 cc",
      "d db Qc",
      "p1 cc",
      "p2 cbr 375",
      "p1 f"
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
        score, equity = preflop_analysis(hole1, trials=10000)
        print(f"P1 Equity:         {equity:.3f}")
        type_flags = {ht: False for ht in hand_types}
        print(f"P1 Hand types:     {type_flags}")
    else:
        equity = simulate_equity(hole1, board, trials=10000)
        print(f"P1 Equity:         {equity:.3f}")
        # Determine hand type flags
        score = safe_evaluate(board + hole1, evaluator)
        rank_class = evaluator.get_rank_class(score)
        hand_name = evaluator.class_to_string(rank_class).lower().replace(' ', '-')
        type_flags = {ht: (ht == hand_name) for ht in hand_types}
        print(f"P1 Hand types:     {type_flags}")
