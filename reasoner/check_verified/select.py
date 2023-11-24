import json

score = 0.7

fread = open("../inference/eval_entailmentbank_task2/lightning_logs/version_0/results_train_verified.json")
fw = open("./task2_select_score{}.txt".format(score), "w")

def find(proof):
    premise_str = proof.split(" -> ")[0]
    premises = premise_str.split(" & ")
    return premises

count = 0
for line in fread:
    info = json.loads(line.strip())
    premises = info["input_fake_seq"]
    idents = info["input_fake_ident"]
    conclusions = info["decode_conclusion"]
    decode_score = info["decode_score_new"]
    stepwise_goal = info["stepwise_goal"]
    goal_premises = find(stepwise_goal)
    for ind, score_ in enumerate(decode_score):
        if score_ >= score:
            premise = premises[ind]
            conclusion = conclusions[ind]
            ident = idents[ind]
            if set(ident) == set(goal_premises):
                continue
            else:
                print(ident)
                print(goal_premises)
                print("$premises$: {}, $conclusion$: {}, $score$: {}\n".format(" !#! ".join(premise), conclusion, score_))
                fw.write("$premises$: {}, $conclusion$: {}, $score$: {}\n".format(" !#! ".join(premise), conclusion, score_))
                count += 1
                break
    if count == 100:
        break
fread.close()
fw.close()"
