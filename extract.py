import toml
from pathlib import Path
from poker import parse_game_log, determine_winner
def get_label(model_player, history):
    board, holes, folded = parse_game_log(history)
    winners = determine_winner(board, holes, folded)
    if ('p'+str(model_player) in winners):
        return True
    return False
dataset = [] 

def extract(cur, num):
    phh_path = Path(cur.format(num))
    doc = toml.loads(phh_path.read_text())
    return doc["actions"]

player = 1 #model represent
for i in range(80): #num of the files in the folder you're labelling
    cur_path = "phh-dataset/data/pluribus/30/{}.phh"
    curlog = extract(cur_path, i)
    #print(curlog)
    win = get_label(1, curlog)
    dataset.append((curlog, win))
print(dataset)