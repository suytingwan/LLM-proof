import sys
sys.path.append('../')

from common import *
from pytorch_lightning.cli import LightningCLI
from datamodule import EntailmentDataModule
from model import EntailmentVerifier

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    cli = CLI(EntailmentVerifier, EntailmentDataModule, save_config_overwrite=True)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
