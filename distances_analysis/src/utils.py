import json
import torch
import pickle
import os
import config

from typing import Tuple, List, Any, Dict
from nltk.corpus import wordnet as wn

universal_pos2wn_pos = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "n":"n", "v":"v", "a":"a", "r":"r"} #looks dumb but allows to generalize later without need for try except clauses or similar approaches



def read_dataset_filtered(path: str, mapping, coarse_to_fine, name) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s = [] 
    clusters =  []
    dict = {}
    with open(path) as f:
        data = json.load(f)
    
    
    correct_same_lemmas = 0
    non_correct_same_lemmas = 0
    correct_diff_lemmas = 0
    non_correct_diff_lemmas = 0
    total_correct = 0
    total_non_correct = 0
    both_same = 0
    none_same = 0 
    correct_same = 0
    non_correct_same = 0

    for sentence_id, sentence_data in data.items():
        #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
        try:
            candidate_clusters = sentence_data["candidate_clusters"]
            keys = list(sentence_data["candidate_clusters"].keys())
            candidate_senses = sentence_data["wn_candidates"]
            lemmas = sentence_data["lemmas"]
            pos = sentence_data["pos_tags"]
        except:
            #creating a dummy dictionary so that the following code can be generalized to both types of dataset instances
            keys = ["1"]
            candidate_clusters = {"1":sentence_data["candidate_clusters"]}
            candidate_senses = {"1":sentence_data["wn_candidates"]}
            lemmas = {1:sentence_data["lemma"]}
            pos = {1: sentence_data["pos"]}
        #candidate clusters is a list of lists, where the inner list is the list of candidate clusters for an instance
        for key in keys:
            current_clusters = candidate_clusters[key]
            #we are only intrested in instances that have at least 2 candidate clusters
            if len(current_clusters)>1:
                #to keep track of wheter the correct or wrong example have the same lemma,
                #  element 0 indicates that the correct example has the same lemma
                #  element 1 indicates that the wrong example has the same lemma
                correct_lemmas_check = [False, False] 

                for cluster in current_clusters:
                    succesfull = False
                    #for all the candidate senses for the current instance
                    for cand in candidate_senses[key]:
                        #check if there are examples in the mapping
                        try: 
                            m = mapping[cand]
                        except:
                            #no examples, succesfull remains False and we move on
                            continue
                        
                        #if the current sense is part of the current cluster
                        if cand in [item for sublist in coarse_to_fine[cluster] for item in sublist]:
                            total_correct += len(m["example_tokens"])#keep track of how many correct examples there are
                            #succesfull = True #for cluster c we have at least one example available (OLD REQUIREMENT)
                            for lemma in m["lemma"]:
                                if lemma == lemmas[int(key)]:
                                    correct_same_lemmas += 1
                                    correct_lemmas_check[0] = True

                                    #for this cluster we have at least one example with the same lemma available
                                    succesfull = True
                                else: correct_diff_lemmas += 1 
                            break
                        else: 
                            total_non_correct += len(m["example_tokens"])
                            for lemma in m["lemma"]:  
                                if lemma == lemmas[int(key)]:
                                    non_correct_same_lemmas += 1
                                    correct_lemmas_check[1] = True
                                else: non_correct_diff_lemmas += 1 
                            pass
                    #if a cluster doesn't have any synset with an example associated to it we move to the next instance 
                    if not succesfull: break

                if succesfull: 
                    data_to_append = {}
                    #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
                    try:
                        try:
                            dict[sentence_id].append(sentence_data["instance_ids"][key])
                        except: 
                            dict[sentence_id] = [sentence_data["instance_ids"][key]]
                        data_to_append["instance_ids"] = {key: sentence_data["instance_ids"][key]}
                        data_to_append["wn_candidates"] =  {key: sentence_data["wn_candidates"][key]}
                        data_to_append["candidate_clusters"] =  {key: sentence_data["candidate_clusters"][key]}
                        data_to_append["senses"] =  {key: sentence_data["senses"][key]}
                        data_to_append["gold_clusters"] =  {key: sentence_data["gold_clusters"][key]}
                        data_to_append["words"] =  sentence_data["words"]
                        data_to_append["lemma"] =  {key: lemmas[int(key)]}
                    
                    except Exception as e:
                        dict[sentence_id] = []
                        key_instance_id = sentence_data["instance_ids"][0]
                        data_to_append["instance_ids"] = {key_instance_id: sentence_data["instance_ids"]}
                        data_to_append["candidate_clusters"] =  {key_instance_id: sentence_data["candidate_clusters"]}
                        data_to_append["gold_clusters"] =  {key_instance_id: [sentence_data["cluster_name"]]}
                        data_to_append["words"] =  sentence_data["example_tokens"]
                        data_to_append["wn_candidates"] = {str(key_instance_id):sentence_data["wn_candidates"]}
                        data_to_append["lemma"] =  {str(key_instance_id): lemmas[int(key)]}
                        

                    sentences_s.append(data_to_append)
                    clusters.append(data_to_append["gold_clusters"])

                    if correct_lemmas_check[0] and not correct_lemmas_check[1]:
                        correct_same += 1
                    if not correct_lemmas_check[0] and correct_lemmas_check[1]:
                        non_correct_same += 1
                    if correct_lemmas_check[0] and correct_lemmas_check[1]:
                        both_same += 1
                    if not correct_lemmas_check[0] and not correct_lemmas_check[1]:
                        none_same += 1
                else: pass
    print("Number of instances where the correct example contains the same lemma:", correct_same_lemmas)
    print("Number of instances where the wrong example contains the same lemma", non_correct_same_lemmas)
    print("Number of instances where the correct example contains a different lemma:", correct_diff_lemmas)
    print("Number of instances where the wrong example contains a different lemma:", non_correct_diff_lemmas)
    print("Number of instances where only the correct example contains the same lemma", correct_same)
    print("Number of instances where only the wrong example contains the same lemma", non_correct_same)
    print("Number of instances where both the correct and wrong example contain the same lemma", both_same)
    print("Number of instances where neither the correct nor the wrong example contain the same lemma", none_same)
    print("Total correct examples:", total_correct)
    print("Total wrong examples:", total_non_correct)
    if name == 'test':
        
        pth = os.path.join(config.data_path, "subsets")

        if not os.path.exists(pth):
            os.makedirs(pth)
        
        with open(pth + "/test_dictionary_lemma.pkl", "wb") as f:
                pickle.dump(dict, f)

    return sentences_s, clusters


def read_examples_mapping(path: str) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s, senses_s = [], []
    mapping = {}
    with open(path) as f:
        file = json.load(f)
    case = 0
    for sentence_id, data in file.items():
            
            #instances in the dataset have 2 different structures, try-except is used to account for this, and process the data accordingly
            try:
                if data['synset_name'] not in mapping.keys():
                    mapping[data["synset_name"]] = {
                        "cluster_name" : data["cluster_name"],
                        "example_tokens" : [data["example_tokens"]],
                        "instance_ids" : [data["instance_ids"]],
                        "lemma" : [data["lemma"]],
                        "pos" : [data["pos"]]
                    }
                else: 
                    mapping[data["synset_name"]]["instance_ids"].append(data["instance_ids"])
                    mapping[data["synset_name"]]["example_tokens"].append(data["example_tokens"])
                    mapping[data["synset_name"]]["lemma"].append(data["lemma"])
                    mapping[data["synset_name"]]["pos"].append(data["pos"])
            except:
                for idx in data["instance_ids"].keys():
                    for current_sense in data["senses"][idx]:
                        if current_sense not in mapping.keys():
                            mapping[current_sense] = {
                                "cluster_name" : data["gold_clusters"][idx],
                                "example_tokens" : [data["words"]],
                                "instance_ids" : [[int(idx)]],
                                "lemma" : [data["lemmas"][int(idx)]],
                                "pos" : [universal_pos2wn_pos[[data["pos_tags"][int(idx)]][0]]]
                            }
                        else: 
                            mapping[current_sense]["instance_ids"].append([int(idx)])
                            mapping[current_sense]["example_tokens"].append(data["words"])
                            mapping[current_sense]["lemma"].append(data["lemmas"][int(idx)])
                            mapping[current_sense]["pos"].append(universal_pos2wn_pos[data["pos_tags"][int(idx)]])

    return mapping





def embed_words(tokenizer, model, words, target_idx):
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    '''
    This function takes as input a list of words and the index of the target word.
    It returns the contextualized embeddings of the first bpe of the word.
    '''

    encoding = tokenizer(words, add_special_tokens=True, is_split_into_words=True, return_tensors="pt")
    tokens = encoding["input_ids"].reshape(1, -1)

    output_idx = encoding.word_to_tokens(target_idx[0]).start
    
    tokens = tokens.to(device)

    # embed the example and get the embeddings of the target word
    with torch.no_grad():
        output = model(tokens)
        last_hidden_state = output.last_hidden_state
        embeddings = last_hidden_state[0, output_idx, :].to(device)
        embeddings = embeddings.squeeze(0)
    
    del output, last_hidden_state

    return embeddings



def load_coarse_to_fine(path):
    with open(path) as f:
        data = json.load(f)
    coarse_senses=list(data.keys())
    dict = {}
    for sense in coarse_senses:
        f = data[sense]
        dict[sense] = f
    return dict


def evaluate(path_predictions, path_labels):

    with open(path_predictions, "rb") as f:
        predictions=pickle.load( f)
    with open(path_labels, "rb") as f:
        total_clusters=pickle.load( f)
    correct = 0
    print(len(predictions), len(total_clusters))
    assert len(predictions)==len(total_clusters)
    for i, pred in enumerate(predictions):
        if pred == total_clusters[i]:
                correct+=1

    print(correct,"/",i+1, "=", correct/len(predictions))
