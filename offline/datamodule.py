import sys
sys.path.append('../')

from common import *
from copy import deepcopy
import itertools
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import json
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from proof import ProofStep, Proof, InvalidProofStep

class EntailmentDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name: str,
        split: str,
        dataset: str,
        max_input_len: int
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )
        # for gpt2
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert split in ("train", "val")
        self.split = split
        self.max_input_len = max_input_len
        self.dataset = dataset
        self.data = self.preprocess(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        premises = [f"$premises$: {premise}" for premise in ex["premises"]]
        conclusion = [f"$conclusion$: {conclusion}" for conclusion in ex["conclusion"]]
        ex["premises"] = premises
        ex["conclusion"] = conclusion
        
        return ex

    def preprocess(self, path:str) -> List[Example]:
        #raise NotImplementedError
        data = []
        num_invalid = 0
        for count, line in enumerate(open(path)):
            ex = json.loads(line.strip())
            hypothesis = ex["hypothesis"]
            context = ex["context"]
            proof_text = ex["proof_gt"].strip()
            proof_id = ex["proof_id"]
            try:
                proof = Proof(
                    context,
                    hypothesis,
                    proof_text,
                    proof_id,
                    strict=True,
                    requires_complete=True
                )
                out_texts = ex["proof_candidates"]
                out_scores = ex["score_candidates"]
                out_premises = []
                out_conclusion = []
                out_mask_scores = []
                for out_text, out_score in zip(out_texts, out_scores):
                    try:
                        step = ProofStep(proof, out_text.strip(";"), strict=False)
                        premises, conclusion = step.serialize()
                        #if step.conclusion_ident == 'int':
                        #    print(out_text)
                        #    print(premises)
                        #    print(conclusion)
                        out_premises.append(premises)
                        out_conclusion.append(conclusion)
                        out_mask_scores.append(1.0)
                    except InvalidProofStep:
                        out_premises.append(out_text)
                        out_conclusion.append(out_text)
                        out_mask_scores.append(0.0)  
                info = deepcopy(ex)
                info["premises"] = out_premises
                info["conclusion"] = out_conclusion
                info["out_mask_scores"] = out_mask_scores
                data.append(info)
            except InvalidProofStep:
                num_invalid += 1
        print("num invalid proofs: ", num_invalid)
        return data

    def collate(self, examples: List[Example]) -> Batch:
        #inp = [ex["premises"] + " " + ex["conclusion"] + self.tokenizer.eos_token for ex in examples]
        # bs = 1
        ex = examples[0]
        inp = [ex["premises"][i] + " " + ex["conclusion"][i] + " " + self.tokenizer.eos_token for i in range(len(ex["premises"]))]
        
        entailment = self.tokenizer(
            inp,
            padding="longest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        input_ids = entailment.input_ids
        labels = input_ids.clone()
        #input_exp = [ex["premises"] + " $conclusion$: " for ex in examples]
        input_exp = [ex["premises"][i] + " $conclusion$: " for i in range(len(ex["premises"]))]
        input_exp_seq = self.tokenizer(
            input_exp,
            padding="longest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        
        input_attention_mask = entailment.attention_mask
        for i in range(input_exp_seq.attention_mask.shape[0]):
            labels[i, :sum(input_exp_seq.attention_mask[i])] = -100 #ignore index

        if self.dataset == "entailmentbank":
            return {
                "premises": ex["premises"],
                "conclusion": ex["conclusion"],
                "input_ids": entailment["input_ids"],
                "attention_mask": entailment["attention_mask"],
                "label": labels,
                "input_seq": input_exp,
                "out_mask_scores": ex["out_mask_scores"],
                "proof_candidates": ex["proof_candidates"],
                "score_candidates": ex["score_candidates"],
                "proof_id": ex["proof_id"],
                "hypothesis": ex["hypothesis"],
                "context": ex["context"],
                "proof_gt": ex["proof_gt"],
                "partial_proof": ex["partial_proof"],
                "stepwise_goal": ex["stepwise_goal"]
            }
        elif self.dataset == "ruletaker":
            return {
                "premises": ex["premises"],
                "conclusion": ex["conclusion"],
                "input_ids": entailment["input_ids"],
                "attention_mask": entailment["attention_mask"],
                "labels": labels,
                "input_seq": input_exp,
                "proof_candidates": ex["proof_candidates"],
                "score_candidates": ex["score_candidates"],
                "out_mask_scores": ex["out_mask_scores"],
                "proof_id": ex["proof_id"],
                "hypothesis": ex["hypothesis"],
                "context": ex["context"],
                "proof_gt": ex["proof_gt"],
                "partial_proof": ex["partial_proof"],
                "stepwise_goal": ex["stepwise_goal"],
                "answer": ex["answer"],
                "depth": ex["depth"],
                "all_proof": ex["all_proof"]
            }


class EntailmentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        path_train: str,
        path_val: str,
        model_name: str,
        batch_size: int,
        num_workers: int,
        max_input_len: int,
        max_output_len: int
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.path_train = path_train
        self.path_val = path_val
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = EntailmentDataset(
                self.path_train,
                self.model_name,
                "train",
                self.dataset,
                self.max_input_len
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = EntailmentDataset(
                self.path_val,
                self.model_name,
                "val",
                self.dataset,
                self.max_input_len
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            pin_memory=True,
            drop_last=False
        )


if __name__ == "__main__":

    path_train = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/decode_sft/prover/eval_entailmentbank_task2/lightning_logs/version_0/results_val.json"
    path_val = "/home/ysuay/codes/LLM+Reason/codes/LLMProofs/data_prompt/entailment_verifier_task2_samples.txt"

    ds_val = EntailmentDataset(
        split="val",
        path=path_train,
        model_name="gpt2",
        dataset="entailmentbank",
        max_input_len=1024)
    print(len(ds_val))
    print(ds_val[0])
    ret = ds_val.collate([ds_val[0], ds_val[1]])
    print(ret)
