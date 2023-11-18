from ast import Tuple
import stat

import wandb
from sparse_autoencoder import SparseAutoencoder, pipeline
from transformers import PreTrainedTokenizerBase

from abc import ABC, abstractmethod, abstractproperty
from transformer_lens import HookedTransformer
from dataclasses import dataclass
import argparse
from src.sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from torch import device
from src.sparse_autoencoder.source_data.text_dataset import GenericTextDataset
from src.sparse_autoencoder.train.sweep_config import SweepParametersRuntime
import torch

@dataclass
class Config:
    model_name:str
    device:device
    expansion_ratio:int = 4

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
        self.activation_name:str
        self.dimensionality:int
        
    @abstractmethod
    def set_dimensionality(self,  model:HookedTransformer) -> None:
        pass
    
    @abstractmethod
    @classmethod
    def get_name(cls) -> str:
        pass
        
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
    
    @property
    def dimensionality(self) -> int:
        return self._dimensionality
    
    def set_dimensionality(self,  model:HookedTransformer) -> None:
        self._dimensionality = model.cfg.d_head
    

class MLPActivation(Activation):
    def __init__(self, layer:int):
        super().__init__(layer)
        self.activation_name = f"blocks.{layer}.mlp.hook_post"
    
    @classmethod
    def get_name(cls, layer:int) -> str:
        return cls(layer).activation_name

    @property
    def dimensionality(self) -> int:
        return self._dimensionality # type: ignore
    
    def set_dimensionality(self,  model:HookedTransformer) -> None:
        self._dimensionality = model.cfg.d_mlp
        
        
        
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

    @property
    def dimensionality(self) -> int:
        return self._dimensionality
    
    def set_dimensionality(self,  model:HookedTransformer) -> None:
        self._dimensionality = model.cfg.d_model



class Trainer:
    def __init__(self, config: Config, data_config:DataConfig, trainer_config:TrainerConfig):
        self.config = config
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.model = HookedTransformer.from_pretrained(config.model_name, device=config.device)
        self.tokenizer: PreTrainedTokenizerBase =  self.model.tokenizer # type: ignore
        self.source_data = GenericTextDataset(tokenizer=self.tokenizer, dataset_path=data_config.dataset_name) # type: ignore
        
    
    def init_autoencoder(self, activation:Activation):
        """
        Initialize the autoencoder with the activation
        """
        self.autoencoder = SparseAutoencoder(
            n_input_features=activation.dimensionality,
            n_learned_features=activation.dimensionality * self.config.expansion_ratio,
            geometric_median_dataset= torch.zeros(activation.dimensionality),
            device=self.config.device,
        )
    
    def run_pipeline(self, activation:Activation):
        """
        Very easy wrapper of the sparse autoencoder pipeline. Good starting point to train the autoencoder.
        Probably bad for doing more complex stuff.
        """
        # this is a bit ugly, maybe we can do better with the activation class (We need now just the name of the hook)
        # TODO: Implement the pipeline
        activation.set_dimensionality(self.model)
        store = TensorActivationStore(self.data_config.max_items, activation.dimensionality, self.config.device)
        
        # set training hyperparameters
        trainig_hyperparameters = SweepParametersRuntime(
            lr=self.trainer_config.lr,
            l1_coefficient=self.trainer_config.l1_coefficient,
            batch_size=self.trainer_config.batch_size,
            adam_beta_1=self.trainer_config.adam_beta_1,
            adam_beta_2=self.trainer_config.adam_beta_2,
            adam_epsilon=self.trainer_config.adam_epsilon,
            adam_weight_decay=self.trainer_config.adam_weight_decay,
        )
        
        # add in a dict all the config
        config_dict = [config.__dit__ for config in [self.config, self.data_config, self.trainer_config]]
        #merge all the config in a single dict
        config_dict = {k: v for d in config_dict for k, v in d.items()}
        
        # init wandb
        wandb.init(project="sparse-autoencoder", config=config_dict)
        
        # init the autoencoder
        self.init_autoencoder(activation)
        
        # run the pipeline
        pipeline(
            src_model=self.model,
            src_model_activation_hook_point=activation.activation_name,
            src_model_activation_layer=activation.layer,
            source_dataset=self.source_data,
            activation_store=store,
            num_activations_before_training=self.data_config.max_items,
            autoencoder=self.autoencoder,
            device=self.config.device,
            max_activations=self.data_config.max_items,
            sweep_parameters=trainig_hyperparameters,
        )