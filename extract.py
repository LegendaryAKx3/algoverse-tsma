import toml
from pathlib import Path
from poker_functions import parse_game_log, determine_winner, evaluate_hand_any_stage
def get_label(model_player, history):
    board, holes = parse_game_log(history)
    winners = determine_winner(board, holes)
    score = evaluate_hand_any_stage(holes['p'+str(model_player)], board)

    if ('p'+str(model_player) in winners):
        return True, score
    return False, score

dataset = [] 

def extract(cur, num):
    phh_path = Path(cur.format(num))
    doc = toml.loads(phh_path.read_text())
    return doc["starting_stacks"], doc["actions"]


player = 1 #model represent
for i in range(80):
    cur_path = "phh-dataset/data/pluribus/30/{}.phh"
    stacks, curlog = extract(cur_path, i)
    win, score = get_label(player, curlog)
    
    dataset.append((stacks, curlog, win, score))

print(dataset)


import json

json_ready = [
    {"stacks": stacks, "actions": actions, "win": win, "score": score}
    for stacks, actions, win, score in dataset
]

with open("labeled_dataset.json", "w") as f:
    json.dump(json_ready, f, indent=2)