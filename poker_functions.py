from itertools import combinations
from collections import Counter, namedtuple

# --- data structures ---
Card = namedtuple('Card', ['rank', 'suit'])

# Map ranks to values for easy comparison
RANK_ORDER = {r: i for i, r in enumerate('23456789TJQKA', start=2)}

# --- parsing utilities ---
def parse_cards(cards_str):
    """
    Given a string like "Jc9h5c" or "Ac2d", return a list of Card(rank, suit).
    """
    cards = []
    for i in range(0, len(cards_str), 2):
        r, s = cards_str[i], cards_str[i+1]
        cards.append(Card(r, s))
    return cards

def parse_game_log(lines):
    """
    Given a list of log lines, returns:
      - board: list of Card
      - hole_cards: dict player_id -> list of Card
    """
    board = []
    hole_cards = {}

    for line in lines:
        parts = line.split()
        # deal hole cards: "d dh p1 5d2d"
        if parts[0]=='d' and parts[1]=='dh':
            player, cards_str = parts[2], parts[3]
            if '?' not in cards_str:
                hole_cards[player] = parse_cards(cards_str)
        # deal board cards: "d db Jc9h5c" or turn/river: single two‑char card
        elif parts[0]=='d' and parts[1]=='db':
            new_cards = parse_cards(parts[2])
            board.extend(new_cards)
        # showdown reveal: "p1 sm Ac2d"
        elif parts[0].startswith('p') and parts[1]=='sm':
           # player, cards_str = parts[0], parts[2]
            hole_cards[player] = parse_cards(cards_str)
        # ignore folding and betting actions
    return board, hole_cards

# --- hand evaluation ---
def rank_counts(cards):
    """Return Counter of ranks in the list of Card."""
    return Counter(card.rank for card in cards)

def is_flush(cards):
    """True if all cards same suit."""
    suits = [c.suit for c in cards]
    return len(set(suits)) == 1

def is_straight(ranks):
    """
    Given sorted list of distinct rank values, check for 5‑card straight.
    Handles the wheel (A‑2‑3‑4‑5) as the lowest straight.
    """
    # handle wheel straight
    if set(ranks[-4:] + [ranks[0]]) == {14, 2, 3, 4, 5}:
        return True, 5
    # sliding window of length 5
    for i in range(len(ranks) - 4):
        window = ranks[i:i+5]
        if window == list(range(window[0], window[0]+5)):
            return True, window[-1]
    return False, None

def evaluate_five(cards):
    """
    Evaluate 5-card hand with simple integer scoring.
    Returns a single integer score where higher is better.
    
    Score ranges:
    9000-9999: Straight Flush
    8000-8999: Four of a Kind  
    7000-7999: Full House
    6000-6999: Flush
    5000-5999: Straight
    4000-4999: Three of a Kind
    3000-3999: Two Pair
    2000-2999: One Pair
    1000-1999: High Card
    """
    ranks = sorted((RANK_ORDER[c.rank] for c in cards), reverse=True)
    rc = rank_counts(cards)
    counts = sorted(rc.values(), reverse=True)
    flush = is_flush(cards)
    straight, top_straight = is_straight(sorted(set(ranks)))
    
    # Straight flush: 9000 + top card rank (suit doesn't matter for ranking)
    if flush and straight:
        return 9000 + top_straight
    
    # Four of a kind: 8000 + quad rank*50 + kicker
    if counts[0] == 4:
        quad_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 4)
        kicker = max(r for r in ranks if r != quad_rank)
        return 8000 + quad_rank * 50 + kicker
    
    # Full house: 7000 + trips*50 + pair
    if counts == [3,2]:
        trip_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 3)
        pair_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 2)
        return 7000 + trip_rank * 50 + pair_rank
    
    # Flush: 6000 + all 5 cards (suit doesn't affect ranking)
    if flush:
        return 6000 + ranks[0]*50 + ranks[1]*4 + ranks[2]*3 + ranks[3]*2 + ranks[4]
    
    # Straight: 5000 + top card
    if straight:
        return 5000 + top_straight
    
    # Three of a kind: 4000 + trips*30 + kicker1*2 + kicker2
    if counts[0] == 3:
        trip_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 3)
        kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
        return 4000 + trip_rank * 30 + kickers[0] * 2 + kickers[1]
    
    # Two pair: 3000 + high_pair*30 + low_pair*2 + kicker
    if counts == [2,2,1]:
        pairs = sorted([RANK_ORDER[r] for r, cnt in rc.items() if cnt == 2], reverse=True)
        kicker = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 1)
        return 3000 + pairs[0] * 30 + pairs[1] * 2 + kicker
    
    # One pair: 2000 + pair*50 + kicker1*4 + kicker2*2 + kicker3
    if counts[0] == 2:
        pair_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 2)
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
        return 2000 + pair_rank * 50 + kickers[0] * 4 + kickers[1] * 2 + kickers[2]
    
    # High card: 1000 + high*50 + second*4 + third*3 + fourth*2 + fifth
    return 1000 + ranks[0]*50 + ranks[1]*4 + ranks[2]*3 + ranks[3]*2 + ranks[4]

def evaluate_hole_cards(hole_cards):
    """Evaluate strength of just 2 hole cards for preflop with simple integer scoring."""
    if len(hole_cards) != 2:
        return (0,)
    
    ranks = sorted([RANK_ORDER[c.rank] for c in hole_cards], reverse=True)
    suited = hole_cards[0].suit == hole_cards[1].suit
    
    if ranks[0] == ranks[1]:  # Pair
        # Pairs: 300-312 (AA=312, KK=311, ..., 22=300)
        base_score = 300 + ranks[0] - 2
        return (base_score,)
    elif suited:  # Suited cards
        # Suited: 200-299 range, subtract gap penalty
        gap_penalty = abs(ranks[0] - ranks[1])
        base_score = 200 + ranks[0] + ranks[1] - gap_penalty
        return (base_score,)
    else:  # Offsuit
        # Offsuit: 100-199 range, subtract larger gap penalty
        gap_penalty = abs(ranks[0] - ranks[1]) * 2
        base_score = 100 + ranks[0] + ranks[1] - gap_penalty
        return (base_score,)

def evaluate_hand_any_stage(hole_cards, board):
    """Evaluate hand strength at any stage with simple integer scoring."""
    if len(hole_cards) != 2:
        return 0
    
    # Preflop: use hole card evaluation
    if len(board) == 0:
        return evaluate_hole_cards(hole_cards)[0]
    
    # Post-flop: combine hole cards with board
    all_cards = hole_cards + board
    
    # If we have 5+ cards, use standard poker hand evaluation
    if len(all_cards) >= 5:
        if len(all_cards) == 5:
            return evaluate_five(all_cards)
        else:
            return best_hand_from_seven(all_cards)[0]
    
    # Less than 5 cards: partial evaluation with simple scoring
    ranks = sorted([RANK_ORDER[c.rank] for c in all_cards], reverse=True)
    rc = rank_counts(all_cards)
    counts = sorted(rc.values(), reverse=True)
    
    # Adjust scoring for partial hands (3-4 cards total)
    if counts[0] >= 3:  # Three of a kind with 3-4 cards
        trip_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt >= 3)
        return 4000 + trip_rank * 30  # Similar to full 5-card trips
    elif counts[0] == 2:
        if len(counts) >= 2 and counts[1] == 2:  # Two pair
            pairs = sorted([RANK_ORDER[r] for r, cnt in rc.items() if cnt == 2], reverse=True)
            return 3000 + pairs[0] * 30 + pairs[1] * 2
        else:  # One pair
            pair_rank = max(RANK_ORDER[r] for r, cnt in rc.items() if cnt == 2)
            kickers = [r for r in ranks if r != pair_rank]
            kicker_sum = sum(kickers[:3])  # Up to 3 kickers
            return 2000 + pair_rank * 50 + kicker_sum
    else:  # High card
        return 1000 + ranks[0] * 50 + sum(ranks[1:4])  # High card + next 3
    

def best_hand_from_seven(seven_cards):
    """
    Find best 5-card hand from 7 cards following Texas Hold'em rules.
    Returns (score, best_5_cards) where score follows standard poker rankings.
    """
    best_score = 0
    best_5 = None
    
    # Try all possible 5-card combinations from the 7 cards
    for combo in combinations(seven_cards, 5):
        score = evaluate_five(combo)
        if score > best_score:
            best_score, best_5 = score, combo
    
    return best_score, best_5

# --- overall winner determination ---
def determine_winner(board, hole_cards):
    """
    Determine winner at any stage of the game.
    Works for preflop (no board), flop (3 cards), turn (4 cards), or river (5 cards).
    """
    scores = {}
    for player, cards in hole_cards.items():
        if len(cards) != 2:  # Skip players without valid hole cards
            continue
        score = evaluate_hand_any_stage(cards, board)
        scores[player] = score

    if not scores:
        return []
    
    # Find max score
    max_score = max(scores.values())
    # Return all players matching it
    winners = [p for p, score in scores.items() if score == max_score]
    return winners

# --- example usage ---
if __name__ == "__main__":
    log = [
      "d dh p1 KcKs",
      "d dh p2 4h2s",
      "d dh p3 Ah2d",
      "d dh p4 9h8h",
      "d dh p5 4cKh",
      "d dh p6 8sJh",
      "p3 f",
      "p4 cbr 200",
      "p5 f",
      "p6 f",
      "p1 cbr 850",
      "p2 f",
      "p4 cc",
      "d db QsTc8d",
      "p1 cc",
      "p4 cc",
      "d db 7h",
      "p1 cbr 1350",
      "p4 cc",
      "d db 7s",
      "p1 cbr 3375",
      "p4 f"
    ]

    board, holes = parse_game_log(log)
    
    print(f"\nWinner: {determine_winner(board, holes)}")
    score = evaluate_hand_any_stage(holes["p1"], board)
    print(f"P1 Score: {score}")
