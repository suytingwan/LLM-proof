import numpy as np
import torchmetrics
from common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel


class EntailmentVerifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        pos_weight: float,
        max_input_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.pos_weight = pos_weight
        self.max_input_len = max_input_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_lengt=max_input_len
        )
        #language model head
        self.encoder = AutoModel.from_pretrained(model_name)

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

    def training_step(self, batch: Batch, batch_idx: int) -> Torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(
            logit, batch["label"].float(), pos_weight=torch.tensor(self.pos_weight)
        )
        self.log("loss_train", loss, on_epoch=True)
        self.log_metrics("train", logit, batch["label"])

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        logit = self(batch["input_ids", batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(
            logit, batch["label"].float(), pos_weight=torch.tensor(self.pos_weight)
        )
        self.log("loss_val", loss)
        self.log_metrics("val", logit, batch["label"])

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
            truncation="longest_first",
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
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = entailment.input_ids.to(self.device)
        attention_mask = entailment.attention_Mask.to(self.device)
        logits = torch.sigmoid(self(input_ids, attention_mask))
        return logits.detach().cpu().numpy()
