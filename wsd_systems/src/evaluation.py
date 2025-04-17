import json
from tqdm import tqdm
import torch
import pickle
import os

# compute accuracy
def test_accuracy(preds, labels):
    tot_n = preds.shape[0]
    correct_n = torch.sum((preds==labels)).item()
    return correct_n/tot_n

# evaluation of fine model which predicts homonym clusters
def fine2cluster_evaluation(model, data):
    cluster2fine_map = json.load(open(f"{data.hparams.data_path}cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("sense_mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("sense_mapping/cluster_id2sense.json", "r"))
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()): ##!
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l]
            fine_candidates = batch["fine_candidates"]
            fine_candidates = [l for item in fine_candidates for l in item]
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = model.predict(batch, fine_candidates, fine_labels, fine_labels_eval)
            # we prepare fine_preds and cluster_candidates...
            fine_preds = fine_preds.tolist()
            cluster_candidates_list = [e for l in batch["cluster_candidates"] for e in l]
            # and then we find the correspondent 'homonym cluster'
            coarse_preds = [] # to be filled
            for fine_pred, cluster_candidates in zip(fine_preds, cluster_candidates_list):
                cluster_found = False
                for cluster_candidate in cluster_candidates:
                    if cluster_found == True: break
                    # we compute the list of all fine sense in the 'cluster_candidate' homonym cluster
                    fine_senses_list = cluster2fine_map[cluster_id2sense[str(cluster_candidate)]] # list of fine senses of a cluster
                    for fine_sense in fine_senses_list:
                        if cluster_found == True: break
                        if fine_id2sense[str(fine_pred)] == fine_sense[0]: # because fine_sense[1] is the gloss of the fine sense
                            coarse_preds.append(cluster_candidate)
                            cluster_found = True
            # we prepare coarse labels
            coarse_labels = torch.tensor(batch["cluster_gold"])
            mask = coarse_labels!=-100
            coarse_labels = coarse_labels[mask]
            coarse_labels = coarse_labels.tolist() # already flattened
            assert len(coarse_preds) == len(coarse_labels)
        
            preds_list += coarse_preds
            labels_list += coarse_labels
        
        assert len(preds_list) == len(labels_list)
        print(f"\nOn a total of {len(preds_list)} samples...")
        ris_accuracy = test_accuracy(torch.tensor(preds_list), torch.tensor(labels_list))
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")

# evaluaytion of coarse and fine models
def base_evaluation(model, data):
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()): ##!
            labels = batch["cluster_gold"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["cluster_gold_eval"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold_eval"]
            preds, labels = model.predict(batch, candidates, labels, labels_eval)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
        
        assert preds_list.shape[0] == labels_list.shape[0]
        print(f"\nOn a total of {preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(preds_list, labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")

# evaluation of a subset of test dataset      
def base_subset_evaluation(model, data):
    items_id_list = []
    with open(data.hparams.data_test) as f: ##!
        data_dict = json.load(f)
    for sentence_id, sentence_data in list(data_dict.items()):
        # old structure
        if type(sentence_data["instance_ids"]) == dict:
            sense_idx_list = list(sentence_data["senses"].keys())
            for sense_idx in sense_idx_list:
                if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"][sense_idx]) == 1:
                    continue
                items_id_list.append(sentence_data["instance_ids"][sense_idx])
        else: # new structure
            if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"]) == 1:
                continue
            items_id_list.append(sentence_id)
    
    pth = os.path.join(data.hparams.data_path, "subsets/test_dictionary_lemma.pkl")

    if not os.path.exists(pth):
        print("test_dictionary_lemma.pkl NOT PRESENT. RUN FIRST OTHER EXPERIMENTS!")
    with open(pth, 'rb') as subset_file: ##!
        loaded_data = pickle.load(subset_file)
    subset_list = []
    for k in loaded_data.keys():
        if len(loaded_data[k]) == 0:
            subset_list.append(k)
        else:
            subset_list += loaded_data[k]
    subset_idx_list = []
    for i in range(len(items_id_list)):
        if items_id_list[i] in subset_list:
            subset_idx_list.append(i)
    
    # PREDICTIONS
    subset_idx_list = torch.tensor(subset_idx_list)
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = torch.tensor([]), torch.tensor([])
        for batch in tqdm(data.test_dataloader()): ##!
            labels = batch["cluster_gold"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold"]
            labels = [label for l in labels for label in l]
            candidates = batch["cluster_candidates"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_candidates"]
            candidates = [l for item in candidates for l in item]
            labels_eval = batch["cluster_gold_eval"] if model.hparams.coarse_or_fine == "coarse" else batch["fine_gold_eval"]
            preds, labels = model.predict(batch, candidates, labels, labels_eval)
            assert preds.shape[0] == labels.shape[0]
            preds_list = torch.cat((preds_list, preds))
            labels_list = torch.cat((labels_list, labels))
            
        subset_preds_list = torch.index_select(preds_list, 0, subset_idx_list)
        subset_labels_list = torch.index_select(labels_list, 0, subset_idx_list)
        assert subset_preds_list.shape[0] == subset_labels_list.shape[0]
        print(f"\nOn a total of {subset_preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(subset_preds_list, subset_labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")
        
# FINE2CLUSTER evaluation of a subset of test dataset    
def fine2cluster_subset_evaluation(model, data):
    items_id_list = []
    with open(data.hparams.data_test) as f: ##!
        data_dict = json.load(f)
    for sentence_id, sentence_data in list(data_dict.items()):
        # old structure
        if type(sentence_data["instance_ids"]) == dict:
            sense_idx_list = list(sentence_data["senses"].keys())
            for sense_idx in sense_idx_list:
                if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"][sense_idx]) == 1:
                    continue
                items_id_list.append(sentence_data["instance_ids"][sense_idx])
        else: # new structure
            if data.hparams.cluster_candidates_filter and len(sentence_data["candidate_clusters"]) == 1:
                continue
            items_id_list.append(sentence_id)
    
    pth = os.path.join(data.hparams.data_path, "subsets/test_dictionary_lemma.pkl")

    if not os.path.exists(pth):
        print("test_dictionary_lemma.pkl NOT PRESENT. RUN FIRST OTHER EXPERIMENTS!")
    with open(pth, 'rb') as subset_file: ##!
        loaded_data = pickle.load(subset_file)
    subset_list = []
    for k in loaded_data.keys():
        if len(loaded_data[k]) == 0:
            subset_list.append(k)
        else:
            subset_list += loaded_data[k]
    subset_idx_list = []
    for i in range(len(items_id_list)):
        if items_id_list[i] in subset_list:
            subset_idx_list.append(i)
    
    # FINE2CLUSTER PREDICTIONS
    # mapping
    cluster2fine_map = json.load(open(f"{data.hparams.data_path}cluster2fine_map.json", "r"))
    fine_id2sense = json.load(open("sense_mapping/fine_id2sense.json", "r"))
    cluster_id2sense = json.load(open("sense_mapping/cluster_id2sense.json", "r"))
    subset_idx_list = torch.tensor(subset_idx_list)
    model.eval()
    with torch.no_grad():
        preds_list, labels_list = [], []
        for batch in tqdm(data.test_dataloader()): ##!
            # we first predict fine senses
            fine_labels = batch["fine_gold"]
            fine_labels = [label for l in fine_labels for label in l]
            fine_candidates = batch["fine_candidates"]
            fine_candidates = [l for item in fine_candidates for l in item]
            fine_labels_eval = batch["fine_gold_eval"]
            fine_preds, _ = model.predict(batch, fine_candidates, fine_labels, fine_labels_eval)
            # we prepare fine_preds and cluster_candidates...
            fine_preds = fine_preds.tolist()
            cluster_candidates_list = [e for l in batch["cluster_candidates"] for e in l]
            # and then we find the correspondent 'homonym cluster'
            coarse_preds = [] # to be filled
            for fine_pred, cluster_candidates in zip(fine_preds, cluster_candidates_list):
                cluster_found = False
                for cluster_candidate in cluster_candidates:
                    if cluster_found == True: break
                    # we compute the list of all fine sense in the 'cluster_candidate' homonym cluster
                    fine_senses_list = cluster2fine_map[cluster_id2sense[str(cluster_candidate)]] # list of fine senses of a cluster
                    for fine_sense in fine_senses_list:
                        if cluster_found == True: break
                        if fine_id2sense[str(fine_pred)] == fine_sense[0]: # because fine_sense[1] is the gloss of the fine sense
                            coarse_preds.append(cluster_candidate)
                            cluster_found = True
            # we prepare coarse labels
            coarse_labels = torch.tensor(batch["cluster_gold"])
            mask = coarse_labels!=-100
            coarse_labels = coarse_labels[mask]
            coarse_labels = coarse_labels.tolist() # already flattened
            assert len(coarse_preds) == len(coarse_labels)
            preds_list += coarse_preds
            labels_list += coarse_labels
            
        subset_preds_list = torch.index_select(torch.tensor(preds_list), 0, subset_idx_list)
        subset_labels_list = torch.index_select(torch.tensor(labels_list), 0, subset_idx_list)
        assert subset_preds_list.shape[0] == subset_labels_list.shape[0]
        print(f"\nOn a total of {subset_preds_list.shape[0]} samples...")
        ris_accuracy = test_accuracy(subset_preds_list, subset_labels_list)
        print()
        print(f"| Accuracy Score for test set:  {round(ris_accuracy, 4)} |")