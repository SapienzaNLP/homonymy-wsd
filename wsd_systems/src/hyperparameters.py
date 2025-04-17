from dataclasses import dataclass

@dataclass
class Hparams:
    
    ## dataloader params
    data_path: str = "data/" # path to the data
    coarse_or_fine: str = "coarse" # coarse-grained or fine-grained task
    data_train: str = f"{data_path}new_split/train.json" # train dataset path
    data_val: str = f"{data_path}new_split/dev.json" # validation dataset path
    data_test: str = f"{data_path}new_split/test.json" # test dataset path
    batch_size: int = 8 # size of the batches
    n_cpu: int = 8 # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    # data filter
    cluster_candidates_filter: bool = False # at least 2 candidates filter
    
    ## train params
    lr: float = 1e-4 # learning rate
    precision: int = 32 # 16 or 32 precision training
    
    ## model params
    encoder: str = "bert" # bert, roberta, deberta, electra
    last_hidden_state: bool = False # use only the last hidden layer or concatenate the last forur
    hidden_dim: int = 512 # hidden dimension of classification head
    dropout: float = 0.1 # dropout value

