from common import *
from pytorch_lightning.cli import LightningCLI
from datamodule import ProofDataModule
from model import EntailmentWriter
from pytorch_lightning.trainer.trainer import Trainer
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("model.stepwise", "data.stepwise")
        parser.link_arguments("data.dataset", "model.dataset")
        parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    cli = CLI(EntailmentWriter, ProofDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
