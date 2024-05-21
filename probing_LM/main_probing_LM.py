import wandb
from argparse import ArgumentParser
from src.data_module import WSD_DataModule
from src.hyperparameters import Hparams
from src.train import train_model
from src.model import WSD_Model
from src.evaluation import base_evaluation, fine2cluster_evaluation, \
                           base_subset_evaluation, fine2cluster_subset_evaluation

import torch
import random
import numpy as np
import wandb
from dataclasses import asdict
import pytorch_lightning as pl

# setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    _ = pl.seed_everything(seed)

def parse_args():
    parser = ArgumentParser()
    # experiments parameters
    parser.add_argument("--mode", type=str, default="train", help="train/eval", required=True)
    parser.add_argument("--type", type=str, default="coarse", help="coarse/fine/fine2cluster/base_subset/fine2cluster_subset", required=True)
    parser.add_argument("--encoder", type=str, default="bert", help="encoder type", required=True)
    parser.add_argument("--model", type=str, default="checkpoints/coarse.ckpt", help="path to model to evaluate", required=False)
    return parser.parse_args()


def main(arguments):
    if arguments.mode == "train":
        wandb.login() # this is the key to paste each time for login: 65a23b5182ca8ce3eb72530af592cf3bfa19de85

        version_name = arguments.type+"_"+arguments.encoder
        with wandb.init(entity="homonyms", project="homonyms", name=version_name, mode="offline"):
            hparams = asdict(Hparams())
            hparams["encoder"] = arguments.encoder
            data = WSD_DataModule(hparams)
            model = WSD_Model(hparams)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            train_model(data, model, experiment_name=version_name, metric_to_monitor="val_accuracy", mode="max", epochs=10, precision=hparams["precision"])

        wandb.finish()
    
    elif arguments.mode == "eval":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hparams = asdict(Hparams())
        hparams["encoder"] = arguments.encoder
        if arguments.type == "coarse" or arguments.type == "fine":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            data = WSD_DataModule(hparams)
            data.setup()
            base_evaluation(model, data)
        
        elif arguments.type == "fine2cluster":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            assert model.hparams.coarse_or_fine == "fine"
            data = WSD_DataModule(hparams)
            data.setup()
            # evaluation on homonym clusters using a fine-grained model
            fine2cluster_evaluation(model, data)
        
        elif arguments.type == "base_subset":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            data = WSD_DataModule(hparams)
            data.setup()
            base_subset_evaluation(model, data)
            
        elif arguments.type == "fine2cluster_subset":
            ckpt = arguments.model
            model = WSD_Model.load_from_checkpoint(ckpt).to(device)
            assert model.hparams.coarse_or_fine == "fine"
            data = WSD_DataModule(hparams)
            data.setup()
            fine2cluster_subset_evaluation(model, data)

if __name__ == '__main__':
    set_seed(99)
    arguments = parse_args()
    main(arguments)