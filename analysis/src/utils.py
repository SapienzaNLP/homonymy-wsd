import pickle
import json
from config import universal_pos2wn_pos, universal_and_wn_to_wn
from typing import List, Dict, Tuple


def read_examples_mapping(path: str) -> Tuple[List[Dict], List[List[List[str]]]]: 
    mapping = {} 
    with open(path) as f: 
        file = json.load(f) 
    for _, data in file.items(): 
             
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
                                "pos" : [universal_and_wn_to_wn[[data["pos_tags"][int(idx)]][0]]] 
                            } 
                        else:  
                            mapping[current_sense]["instance_ids"].append([int(idx)]) 
                            mapping[current_sense]["example_tokens"].append(data["words"]) 
                            mapping[current_sense]["lemma"].append(data["lemmas"][int(idx)]) 
                            mapping[current_sense]["pos"].append(universal_and_wn_to_wn[data["pos_tags"][int(idx)]]) 
 
    return mapping


def map_synset_to_examples_new_split(sentences_s):
    synset_to_examples = {}
    for sentence in sentences_s:
        if len(list(sentence["senses"].values())) > 1:
            print("MORE THAN ONE SENSE")
        key = list(sentence["senses"].keys())[0]
        sense = sentence["senses"][key]
        entry = (sentence["words"], 
                 sentence["instance_ids"][key], 
                 sentence["gold_clusters"][key], 
                 sentence["lemmas"][key], 
                 sentence["pos"][key])
        if sense not in synset_to_examples:
            synset_to_examples[sense] = [entry] 
        else:
            synset_to_examples[sense].append(entry)
    return synset_to_examples


def merge_examples(old_examples, new_examples):
    for key, value in new_examples.items():
        if key in old_examples:
            old_examples[key].extend(value)
        else:
            old_examples[key] = value 
    return old_examples


