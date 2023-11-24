import json

score = 0.9
fread = open("../inference/eval_entailmentbank_task2/lightning_logs/version_0/results_train_verified.json")
def find(proof):
    premise_str = proof.split(" -> ")[0]
    premises = premise_str.split(" & ")
    return premises

count = 0
total = 0
for line in fread:
    info = json.loads(line.strip())
    stepwise_goal = info["stepwise_goal"]
    idents = info["input_fake_ident"]
    decode_score = info["decode_score_new"]
    goal_premises = find(stepwise_goal)
    for ind, score_ in enumerate(decode_score):
        ident = idents[ind]
        if set(ident) == set(goal_premises):
            continue
        else:
            total += 1
            if score_ >= score:
                count += 1
print(count)
print(total)
