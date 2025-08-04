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
            player, cards_str = parts[0], parts[2]
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
    Given exactly 5 Card objects, return a tuple:
      (category_code, tiebreaker1, tiebreaker2, ...)
    Higher tuples mean better hands.
    
    Categories (highest to lowest):
      9 = Straight Flush
      8 = Four of a Kind
      7 = Full House
      6 = Flush
      5 = Straight
      4 = Three of a Kind
      3 = Two Pair
      2 = One Pair
      1 = High Card
    """
    ranks = sorted((RANK_ORDER[c.rank] for c in cards), reverse=True)
    rc = rank_counts(cards)
    counts = sorted(rc.values(), reverse=True)    # e.g. [3,2] for full house
    distinct = sorted(set(ranks), reverse=True)
    flush = is_flush(cards)
    straight, top_straight = is_straight(sorted(set(ranks)))
    
    # straight flush
    if flush and straight:
        return (9, top_straight)
    # four of a kind
    if counts[0] == 4:
        quad_rank = [r for r, cnt in rc.items() if cnt==4][0]
        kicker = max(r for r in ranks if r != RANK_ORDER[quad_rank])
        return (8, RANK_ORDER[quad_rank], kicker)
    # full house
    if counts == [3,2]:
        trip = [r for r,cnt in rc.items() if cnt==3][0]
        pair = [r for r,cnt in rc.items() if cnt==2][0]
        return (7, RANK_ORDER[trip], RANK_ORDER[pair])
    # flush
    if flush:
        return (6, *ranks)
    # straight
    if straight:
        return (5, top_straight)
    # three of a kind
    if counts[0] == 3:
        trip = [r for r,cnt in rc.items() if cnt==3][0]
        kickers = sorted((RANK_ORDER[r] for r,cnt in rc.items() if cnt==1), reverse=True)
        return (4, RANK_ORDER[trip], *kickers)
    # two pair
    if counts == [2,2,1]:
        pairs = sorted((RANK_ORDER[r] for r,cnt in rc.items() if cnt==2), reverse=True)
        kicker = [RANK_ORDER[r] for r,cnt in rc.items() if cnt==1][0]
        return (3, *pairs, kicker)
    # one pair
    if counts[0] == 2:
        pair = [r for r,cnt in rc.items() if cnt==2][0]
        kickers = sorted((RANK_ORDER[r] for r,cnt in rc.items() if cnt==1), reverse=True)
        return (2, RANK_ORDER[pair], *kickers)
    # high card
    return (1, *ranks)

def evaluate_hole_cards(hole_cards):
    """Evaluate strength of just 2 hole cards for preflop."""
    if len(hole_cards) != 2:
        return (0,)
    
    ranks = sorted([RANK_ORDER[c.rank] for c in hole_cards], reverse=True)
    suited = hole_cards[0].suit == hole_cards[1].suit
    
    # Pair
    if ranks[0] == ranks[1]:
        return (3, ranks[0])
    # Suited cards
    elif suited:
        return (2, ranks[0], ranks[1])
    # Offsuit
    else:
        return (1, ranks[0], ranks[1])

def evaluate_hand_any_stage(hole_cards, board):
    """Evaluate hand strength at any stage of the game."""
    # Preflop: just evaluate hole cards
    if len(board) == 0:
        return evaluate_hole_cards(hole_cards)
    
    # Post-flop: combine hole cards with board
    all_cards = hole_cards + board
    
    # If we have 5+ cards, use standard poker hand evaluation
    if len(all_cards) >= 5:
        if len(all_cards) == 5:
            return evaluate_five(all_cards)
        else:
            return best_hand_from_seven(all_cards)[0]
    

def best_hand_from_seven(seven_cards):
    """Find best 5-card hand from 7 cards."""
    best = None
    best_5 = None
    for combo in combinations(seven_cards, 5):
        rank = evaluate_five(combo)
        if best is None or rank > best:
            best, best_5 = rank, combo
    return best, best_5

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
      "d dh p1 7s5s",
      "d dh p2 Jc3c",
      "d dh p3 8c2s",
      "d dh p4 KsJd",
      "d dh p5 Td5d",
      "p3 f",
      "p4 cbr 400000",
      "p5 f",
      "p1 f",
      "p2 cc",
      "d db 8dAdQc",
      "p2 cc",
      "p4 cbr 350000",
      "p2 f"
    ]

    board, holes = parse_game_log(log)
    
    # Show winners at different stages
    print(f"Winner: {determine_winner(board, holes)}")
