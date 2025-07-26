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
      - folded: set of player_ids who folded
    """
    board = []
    hole_cards = {}
    folded = set()

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
        # player folds: "p2 f"
        elif parts[0].startswith('p') and parts[1]=='f':
            folded.add(parts[0])
        # showdown reveal: "p1 sm Ac2d"
        elif parts[0].startswith('p') and parts[1]=='sm':
            player, cards_str = parts[0], parts[2]
            hole_cards[player] = parse_cards(cards_str)
        # else: ignore betting actions
    return board, hole_cards, folded

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

def best_hand_from_seven(seven_cards):
    """
    From 7 cards, try all 5‑card combos and pick the one with highest evaluate_five().
    Returns (best_rank_tuple, best_5card_list).
    """
    best = None
    best_5 = None
    for combo in combinations(seven_cards, 5):
        rank = evaluate_five(combo)
        if best is None or rank > best:
            best, best_5 = rank, combo
    return best, best_5

# --- overall winner determination ---
def determine_winner(board, hole_cards, folded):
    """
    Given board (list of Card), hole_cards dict, and folded set,
    return list of winning player_ids (tie yields multiple).
    """
    scores = {}
    for player, cards in hole_cards.items():
        if player in folded or len(cards) != 2:
            continue
        seven = cards + board
        score, best5 = best_hand_from_seven(seven)
        scores[player] = (score, best5)

    # find max score
    max_score = max(score for score, _ in scores.values())
    # return all players matching it
    winners = [p for p, (score, _) in scores.items() if score == max_score]
    return winners

# --- example usage ---
if __name__ == "__main__":
    log = [
      "d dh p1 5d2d",  
      "d dh p2 ????",  
      "d dh p3 Jd6h",  
      "p3 cbr 7000",
      "p1 cbr 23000",
      "p2 f",
      "p3 cc",
      "d db Jc9h5c",
      "p1 cbr 35000",
      "p3 cc",
      "d db 4h",
      "p1 cbr 90000",
      "p3 cbr 232600",
      "p1 cbr 1067100",
      "p3 cc",
      "p1 sm Ac2d",
      "p3 sm 7h6h",
      "d db Jh",
    ]

    board, holes, folded = parse_game_log(log)
    winners = determine_winner(board, holes, folded)
    print("Board:", [''.join((c.rank,c.suit)) for c in board])
    for p, cards in holes.items():
        print(f"{p} hole cards:", [''.join((c.rank,c.suit)) for c in cards])
    print("Folded:", folded)
    print("Winner(s):", winners)
