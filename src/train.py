from ast import Tuple
import stat
from sparse_autoencoder import SparseAutoencoder, pipeline
from transformers import PreTrainedTokenizerBase

from abc import ABC, abstractmethod
from transformer_lens import HookedTransformer
from dataclasses import dataclass
import argparse

from src.sparse_autoencoder.source_data.text_dataset import GenericTextDataset


@dataclass
class Config:
    model_name:str
    device:str

@dataclass
class DataConfig:
    dataset_name:str
    max_items:int
    total_training_tokens:int
    
@dataclass
class TrainerConfig:
    lr:float = 1e-3
    l1_coefficient:float = 1e-3 #Large lead to sparsity
    batch_size:int = 4096
    
    # Adam standard (better do not change)
    adam_beta_1:float = 0.9
    adam_beta_2:float = 0.999
    adam_epsilon:float = 1e-8
    adam_weight_decay:float = 0.0
    
class Parser:
    """Parser for the command line arguments, which are used to configure the training process.
    Simple usage, in the main function of the training script:
    config, data_config, trainer_config = Parser.get_config()
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model_name', type=str, default='gpt2')
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--dataset_name', type=str, default='wikitext-2')
        self.parser.add_argument('--max_items', type=int, default=1000)
        self.parser.add_argument('--total_training_tokens', type=int, default=1000000)
        
    @classmethod
    def get_config(cls) -> tuple[Config, DataConfig, TrainerConfig]:
        """Get the config from the parsed arguments."""
        parser = cls()
        return parser.parse_and_configure()
    
    def add_argument(self, *args, **kwargs) -> None:
        """Add an argument to the parser."""
        self.parser.add_argument(*args, **kwargs)
        
    def parse_args(self) -> argparse.Namespace:
        """Parse the arguments."""
        return self.parser.parse_args()
    
    def parse_and_configure(self) -> tuple[Config, DataConfig, TrainerConfig]:
        """Get the config from the parsed arguments."""
        args = self.parse_args()
        return Config(args.model_name, args.device), DataConfig(args.dataset_name, args.max_items, args.total_training_tokens), TrainerConfig()



class Activation(ABC):
    """
    Abstract class for the different activations. We need to know the name of the hook in the model, but in future we may need more information or 
    we may want to do some checks on the activation or store more information.
    """
    def __init__(self, layer:int):
        self.layer = layer
        self.activation_name = None
    
    @abstractmethod
    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError("Abstract Method")
        
class HeadActivation(Activation):
    def __init__(self, layer:int, head:int, subtype:str):
        super().__init__(layer)
        self.head = head
        self.subtype = subtype
        # TODO. model in transformer_lens do not have hooks name for single head. If we want to train a 
        # TODO. autoencoder on a single head, we need to add the hooks to the model (model.add_hooks() or something like that)
        raise NotImplementedError("TODO")

    def __assert__(self):
        assert self.subtype in ["k","q","v","out"], f"Subtype {self.subtype} is not valid. Must be one of ['k','q','v','out']"

    @classmethod
    def get_name(cls, layer:int, head:int, subtype:str) -> str:
        raise NotImplementedError("TODO")

class MLPActivation(Activation):
    def __init__(self, layer:int):
        super().__init__(layer)
        self.activation_name = f"blocks.{layer}.mlp.hook_post"
    
    @classmethod
    def get_name(cls, layer:int) -> str:
        return cls(layer).activation_name

        
class ResidualStreamActivation(Activation):
    def __init__(self, layer:int, subtype:str):
        super().__init__(layer)
        self.subtype = subtype
        self.activation_name = f"blocks.{layer}.hook_resid_{subtype}"
        self.__assert__()
    
    def __assert__(self):
        assert self.subtype in ["pre","post"], f"Subtype {self.subtype} is not valid. Must be one of ['pre','post']"
        
    @classmethod
    def get_name(cls, layer:int, subtype:str) -> str:
        return cls(layer, subtype).activation_name



class Trainer:
    def __init__(self, config: Config, data_config:DataConfig, trainer_config:TrainerConfig):
        self.config = config
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.model = HookedTransformer.from_pretrained(config.model_name, device=config.device)
        self.tokenizer: PreTrainedTokenizerBase =  self.model.tokenizer # type: ignore
        self.source_data = GenericTextDataset(tokenizer=self.tokenizer, dataset_path=data_config.dataset_name) # type: ignore
        
    
    def run_pipeline(self, activation:Activation):
        """
        Very easy wrapper of the sparse autoencoder pipeline. Good starting point to train the autoencoder.
        Probably bad for doing more complex stuff.
        """
        activation_name = activation.activation_name # this is a bit ugly, maybe we can do better with the activation class (We need now just the name of the hook)
        # TODO: Implement the pipeline