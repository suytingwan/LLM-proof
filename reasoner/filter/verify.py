import torch
import json, tqdm
import copy
from transformers import AutoTokenizer, T5EncoderModel

device = "cuda:0" if torch.cuda.is_available else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./vera")
model = T5EncoderModel.from_pretrained("./vera")

model.D = model.shared.embedding_dim
linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))

model = model.to(device)
linear = linear.to(device)
model.eval()
t = model.shared.weight[32097, 0].item()

fread = open("../inference/eval_entailmentbank_task1/lightning_logs/version_0/results_train.json")
fw = open("../inference/eval_entailmentbank_task1/lightning_logs/version_0/results_train_verified.json", 'w')

for line in fread:
    info = json.loads(line.strip())
    decode_score_new = []
    for fake_seq, conclusion in zip(info["input_fake_seq"], info["decode_conclusion"]):
        statement = "Because " + ", and".join(fake_seq)
        statement += "Therefore, " + conclusion
        input_ids = tokenizer.batch_encode_plus([statement], return_tensors='pt', padding='longest', truncation='longest_first', max_length=512).input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output = model(input_ids)
            last_hidden_state = output.last_hidden_state
            hidden = last_hidden_state[:, -1, :]
            logit = linear(hidden).squeeze(-1)
            logit_calibrated = logit / t
            score_calibrated = logit_calibrated.sigmoid()
            #import pdb
            #pdb.set_trace()
            decode_score_new.append(score_calibrated.detach().item())

    info["decode_score_new"] = decode_score_new
    fw.write(json.dumps(info) + '\n')

fread.close()
fw.close()
