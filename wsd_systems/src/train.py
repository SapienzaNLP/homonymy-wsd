import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# standard training pipeline which leverage the power of pytorch-lightning Trainer class
def train_model(data, model, experiment_name, metric_to_monitor, mode, epochs, precision):
    logger =  WandbLogger() # wandb logger
    logger.experiment.watch(model, log = None, log_freq = 100000)
    
    # we also want to save the best checkpoints during training
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1, monitor=metric_to_monitor, mode=mode, dirpath="checkpoints",
    filename=experiment_name + "-{epoch:02d}-{val_accuracy:.4f}", verbose=True)
    
    n_gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(
        logger=logger, max_epochs=epochs, log_every_n_steps=1, gpus=n_gpus,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0, precision=precision, gradient_clip_algorithm="norm", gradient_clip_val=1.0, accumulate_grad_batches=2,
        )
    trainer.fit(model, data)