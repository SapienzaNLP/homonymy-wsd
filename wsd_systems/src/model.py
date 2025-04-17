import torch
from torch import optim
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel
import json
import wandb
import random
from src.evaluation import test_accuracy
import os

def get_num_senses(coarse_or_fine, data_path="data"):
    pth = os.path.join(data_path, "cluster2fine_map.json")
    d = json.load(open(pth, "r"))
    ris = 0
    if coarse_or_fine == "coarse":
        ris = len(list(d.keys()))
    else:
        all_senses_list = []
        for k in d.keys():
            for fine_s in d[k]:
                all_senses_list.append(fine_s[0])
        ris = len(list(set(all_senses_list)))
    return ris

class WSD_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(WSD_Model, self).__init__()
        self.save_hyperparameters(hparams)
        print(f"___________ {self.hparams.coarse_or_fine} MODEL ___________\n")
        if self.hparams.encoder == "bert": self.encoder = AutoModel.from_pretrained("bert-large-cased")
        elif self.hparams.encoder == "roberta": self.encoder = AutoModel.from_pretrained("roberta-large")
        elif self.hparams.encoder == "deberta": self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        elif self.hparams.encoder == "electra": self.encoder = AutoModel.from_pretrained("google/electra-large-discriminator")
        
        # we set all parameters to be not trainable
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # classification HEAD
        n = 1 if self.hparams.last_hidden_state else 4
        self.batch_norm = nn.BatchNorm1d(n*1024)
        self.hidden_MLP = nn.Linear(n*1024, self.hparams.hidden_dim, bias=True)
        self.act_fun = nn.SiLU(inplace=True) # Swish activation function
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.num_senses = get_num_senses(self.hparams.coarse_or_fine, self.hparams.data_path) # sense inventory size
        self.classifier = nn.Linear(self.hparams.hidden_dim, self.num_senses, bias=True) # final linear projection with bias
        
        # mapping from 'id' to name of the 'sense'
        self.id2sense = json.load(open("sense_mapping/cluster_id2sense.json", "r")) if self.hparams.coarse_or_fine == "coarse" else json.load(open("sense_mapping/fine_id2sense.json", "r"))
        # debug infos
        self.debug = False
        # we need to set this field depending on wich batch we want to use (RAM GPU size issues)
        self.gpu_or_cpu = "cuda" # "cuda" or "cpu"
        # if the precision is set to 16 bit, the cross entropy should have GPU tensors!
        assert self.hparams.precision == 32 or self.gpu_or_cpu == "cuda"
    
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
               
    def forward(self, batch):
        text = batch["inputs"].to(self.device)
        # we take only the last hidden layer
        if self.hparams.last_hidden_state:
            embed_text = self.encoder(**text)
            embed_text = embed_text.last_hidden_state
        else: # we take the hidden representation of the last four layers of each token
            embed_text = self.encoder(**text, output_hidden_states=True) # ["input_ids"].to(self.device), attention_mask=text["attention_mask"].to(self.device), token_type_ids=text["token_type_ids"].to(self.device), output_hidden_states=True)
            embed_text = torch.cat(embed_text.hidden_states[-4:], dim=-1) # concatenation --> (batch, seq_len, 1024*4=4096)
        
        encoder_output = embed_text.view(-1, embed_text.shape[-1]) # flattened embeddings --> (batch*seq_len, 4096)
        encoder_output_norm = self.batch_norm(encoder_output)
        hidden_output = self.dropout(self.act_fun(self.hidden_MLP(encoder_output_norm)))
        
        return self.classifier(hidden_output) # (batch*seq_len, num_senses)

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
        }
    
    def training_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        outputs = self(batch)
        outputs = outputs.to(self.gpu_or_cpu)
        labels = torch.tensor(labels).view(-1) # (batch*seq_len)
        labels = labels.to(self.gpu_or_cpu)
        
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(outputs, labels)
        self.log_dict({"loss" : loss})
        return {'loss': loss}

    def predict(self, batch, filter_ids, labels, labels_eval):
        with torch.no_grad():
            outputs = self(batch)
            outputs = outputs.cpu().detach() # we don't need to compute gradient
            # we first need to filter-out the outputs relative to the labels != -100 (the one we are interested in)
            labels = torch.tensor(labels)
            mask = labels!=-100
            labels_filter = mask.nonzero().view(-1) # returns the indeces to keep
            filter_outputs = torch.index_select(outputs, 0, labels_filter)
            ris = []
            for i in range(len(filter_outputs)):
                # predict only over the possible candidates
                filter_ids_i = torch.tensor(filter_ids[i])
                candidates_pred = torch.index_select(filter_outputs[i], 0, filter_ids_i)
                best_prediction = torch.argmax(candidates_pred, dim=0)
                ris.append(filter_ids_i[best_prediction.item()])
            
            # 'labels_eval' are the multi-label labels we need for evaluate accuracy
            labels_eval = [e for l in labels_eval for e in l]
            assert len(ris) == len(labels_eval)
            ris_labels = []
            for pred, gold_labels in zip(ris, labels_eval):
                if pred in gold_labels: # if pred is one of the gold labels
                    ris_labels.append(pred)
                else: # wrong prediction
                    ris_labels.append(gold_labels[0])
            return torch.tensor(ris), torch.tensor(ris_labels)
    
    def validation_step(self, batch, batch_idx):
        labels = batch["cluster_gold"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
        outputs = self(batch)
        outputs = outputs.to(self.gpu_or_cpu)
        labels = torch.tensor(labels).view(-1)
        labels = labels.to(self.gpu_or_cpu)
        # LOSS
        cross_entropy_loss = nn.CrossEntropyLoss()
        val_loss = cross_entropy_loss(outputs, labels)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # ACCURACY
        filter_ids = batch["cluster_candidates"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
        filter_ids = [l for item in filter_ids for l in item] # flattening
        labels = labels.tolist()
        labels_eval = batch["cluster_gold_eval"] if self.hparams.coarse_or_fine == "coarse" else batch["fine_gold_eval"]
        preds, labels = self.predict(batch, filter_ids, labels, labels_eval) # output 'preds' and 'labels' both tensors
        assert preds.shape[0] == labels.shape[0]
        return {"preds": preds, "labels" : labels}
        
        # DEBUG info for one random item of the batch
        if self.debug:
            idx = random.randint(0, labels.shape[0]-1)
            filter_ids = [l for item in filter_ids for l in item]
            candidates = [self.id2sense[str(e)] for e in filter_ids[idx]]
            gold = self.id2sense[str(labels[idx].item())]
            pred = self.id2sense[str(preds[idx].item())]
            debug_infos = str(candidates) + " | " + str(gold) + " | " + str(pred)
            return {"preds": preds, "labels" : labels, "debug_infos" : debug_infos}
    
    # at the end of the epoch, we log the DEBUG infos of each batch
    def validation_epoch_end(self, outputs):
        # log average validation accuracy
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        assert preds.shape[0] == labels.shape[0]
        val_accuracy = test_accuracy(preds, labels)
        self.log("val_accuracy", val_accuracy)
        
        if self.debug:
            c = [str(i+1) for i in range(len(outputs))]
            data = []
            for i in range(len(outputs)):
                data.append(outputs[i]["debug_infos"])
            self.logger.experiment.log({f"debug_infos_{self.current_epoch}": wandb.Table(columns=c, data=[data])})