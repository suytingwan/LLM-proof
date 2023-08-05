import numpy as np
import torchmetrics
from common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


class EntailmentVerifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        topk: int,
        max_input_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.topk = topk
        self.max_input_len = max_input_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=max_input_len
        )
        # for gpt
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #language model head
        self.seq2seq = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.seq2seq(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def generate_conclusion(self, input_text: List[str]) -> Tuple[List[str], List[float]]:
        assert self.trainer is not None
        input = self.tokenizer(
            input_text,
            padding="longest",
            max_length=self.trainer.datamodule.max_input_len,
            truncation=True,
            return_tensors="pt",
        )
        output = self.seq2seq.generate(
            input_ids=input.input_ids.to(self.device, non_blocking=True),
            attention_mask=input.attention_mask.to(self.device, non_blocking=True),
            max_length=self.trainer.datamodule.max_output_len,
            num_beams=self.num_beams,
            num_return_sequences=1,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        output_text = [output_text_.strip().split("$conclusion$: ")[-1] for output_text_ in output_text]
        scores = output.sequences_scores.detach().exp().tolist()
        return output_text, scores

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        self.log("loss_train", loss, on_epoch=True)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        loss = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        self.log("loss_val", loss)

        pred, score = self.generate_conclusion(batch["input_seq"])
        return pred, score, batch["premises"], batch["conclusion"]

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())
                // self.trainer.accumulate_grad_batches
            )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )

    def score(self, premises: List[str], conclusion: str) -> float:
        entailment = self.tokenizer(
            ". ".join(premises) + ".",
            conclusion,
            padding="logest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = entailment.input_ids.to(self.device)
        attention_mask = entailment.attention_mask.to(self.device)
        logit = torch.sigmoid(self(input_ids, attention_mask))
        return logit.detach().item()

    def batch_score(
        self, premises_batch: List[List[str]], conclusion_batch: List[str]
    ) -> Any:
        assert len(premises_batch) == len(conclusion_batch)
        if len(premises_batch) == 0:
            return np.array([])
        entailment = self.tokenizer(
            [". ".join(premises) + "." for premises in premises_batch],
            conclusion_batch,
            padding="longest",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = entailment.input_ids.to(self.device)
        attention_mask = entailment.attention_Mask.to(self.device)
        logits = torch.sigmoid(self(input_ids, attention_mask))
        return logits.detach().cpu().numpy()

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        results = []
        for out in outputs:
            for pred, score, premises, conclusion in zip(*out):
                results.append(
                    {
                        "pred": pred,
                        "score": score,
                        "premises": premises,
                        "conclusion": conclusion
                    }
                )

        assert self.trainer is not None
        if self.logger is not None and self.trainer.log_dir is not None:
            json_path = os.path.join(self.trainer.log_dir, "results.json")
            json.dump(results, open(json_path, "wt"))
            print(f"validation results saved to {json_path}")

