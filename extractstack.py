import toml
import json
from pathlib import Path
import re
import os
import ast
from poker_functions import evaluate_hand_any_stage, parse_game_log, determine_winner
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
    return doc['starting_stacks'], doc["actions"]

fir = "phh-dataset/data/pluribus/{}"
arr = "phh-dataset/data/pluribus/{}/{}.phh"
player = 1 #model represent
cnt = 0

for i in range(30, 118):
    if (Path(fir.format(i)).exists() == True):
        for j in range(500): #num of the files in the folder you're labelling
            cur_path = arr.format(str(i), j)
            if (Path(cur_path).exists() == False):
                continue
            stacks, curlog = extract(cur_path, i)
            win, score = get_label(player, curlog)
            dataset.append((stacks, curlog, win, score))
    if (Path(fir.format(str(i)+'b')).exists() == True):
        for j in range(500): #num of the files in the folder you're labelling
            cur_path = arr.format(str(i)+'b', j)
            if (Path(cur_path).exists() == False):
                continue
            stacks, curlog = extract(cur_path, i)
            win, score = get_label(player, curlog)
            dataset.append((stacks, curlog, win, score))
fir = "phh-dataset/data/wsop/2023/43/5/{}"
for i in range(0, 4):
    for j in range(0, 80):
        for k in range(0, 80):
            curstr = "0"+str(i)+"-"+str(j)+"-"+str(k)+".phh"
            temp = fir.format(curstr)
            
            if (Path(temp).exists() == True):
                #print(temp)
                stacks, curlog = extract(temp, i)
                if (len(curlog[0]) > 13):
                    continue
                stacks, curlog = extract(temp, i)
                win, score = get_label(player, curlog)
                dataset.append((stacks, curlog, win, score))

#print(os.listdir(r"phh-dataset\data\handhq\ABS-2009-07-01_2009-07-23_50NLH_OBFU\0.5"))
"""actions_only = load_actions(
    "phh-dataset/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs"
)"""
#print(get_label(1, actions_only[0]))
print(len(dataset))
json_ready = [
    {"stacks": stacks, "actions": actions, "win": win, "score": score}
    for stacks, actions, win, score in dataset
]

with open("labeled_dataset.json", "w") as f:
    json.dump(json_ready, f, indent=2)