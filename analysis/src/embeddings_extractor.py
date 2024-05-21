import torch
from transformers import AutoTokenizer, AutoModel
import json
import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import config
import random
from utils import map_synset_to_examples_new_split, read_examples_mapping, merge_examples


'''
This file contains the classes and functions to extract lemmas and cluster mappings
from the dataset.
'''

def open_dataset(path):
    '''
    Open the dataset from the json file.

    Args:
        path: the path to the json file

    Returns:
        the dataset
    '''
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset

def map_synset_to_examples(dataset, synset_to_examples_map={}):
    '''
    This function takes as input a dictionary (dataset) that contains
    the examples of the dataset and maps each synset to the examples.
    The format for the input dataset is the same as the one used in the
    "mapped_semcord.json" file.
    '''
    new = 0
    old = 0
    for key, value in dataset.items():
        targets = list(value["senses"].keys())
        synsets = list(value["senses"].values())
        lemmas = value["lemmas"]
        pos_tags = value["pos_tags"]
        num_elements = len(targets)

        for i in range(num_elements):
            target_idx = int(targets[i])
            synset = synsets[i][0]
            cluster = list(value["gold_clusters"].values())[i][0]
            if synset not in synset_to_examples_map:
                synset_to_examples_map[synset] = [(value["words"], [target_idx], cluster, lemmas[target_idx], config.universal_pos2wn_pos[pos_tags[target_idx]])]
                new += 1
            else:
                synset_to_examples_map[synset].append((value["words"], [target_idx], cluster, lemmas[target_idx], config.universal_pos2wn_pos[pos_tags[target_idx]]))
                old += 1
    
    print("New:", new)
    print("Old:", old)
    return synset_to_examples_map

def map_synset_to_wn_examples(dataset, synset_to_examples_map={}):
    '''
    This function takes as input a dictionary (dataset) that contains
    the examples of the dataset and maps each synset to the examples.
    The format is the same as the one used in the "mapped_wn_examples.json"
    file.
    '''
    new = 0
    old = 0
    for row in dataset:
        target_idx = row["instance_ids"]
        target_idx = [int(idx) for idx in range(target_idx[0], target_idx[1])]
        synset = row["synset_name"]
        # example = " ".join(row["example_tokens"])
        cluster = row["cluster_name"]
        if synset not in synset_to_examples_map:
            synset_to_examples_map[synset] = [(row["example_tokens"], target_idx, cluster, row["lemma"], row["pos"])]
            new += 1
        else:
            synset_to_examples_map[synset].append((row["example_tokens"], target_idx, cluster, row["lemma"], row["pos"]))
            old += 1
            
    print("New:", new)
    print("Old:", old)
    return synset_to_examples_map


def embed_words(tokenizer, model, words, target_idx):
    '''
    This function takes as input a list of words and the index of the target word.
    It returns the contextualized embeddings of the first bpe of the word.
    '''

    encoding = tokenizer(words, add_special_tokens=True, is_split_into_words=True, return_tensors="pt")
    tokens = encoding["input_ids"].reshape(1, -1)

    output_idx = encoding.word_to_tokens(target_idx[0]).start
    
    tokens = tokens.to(config.device)

    # embed the example and get the embeddings of the target word
    with torch.no_grad():
        output = model(tokens)
        last_hidden_state = output.last_hidden_state
        embeddings = last_hidden_state[0, output_idx, :].to("cpu")
        embeddings = embeddings.squeeze(0)
    
    del output, last_hidden_state

    return embeddings

def map_synsets_to_embeddings(tokenizer, model, synset_to_examples_map, save_path = None):
    '''
    This function embeds the examples of the dataset using the model passed as input and then builds 
    a dictionary where the keys are the synsets and the values are the embeddings of the examples of that
    synset.
    '''
    synset_to_embeddings = {}
    # embed the examples and save them in a dictionary
    bar = tqdm.tqdm(total=len(synset_to_examples_map))
    for synset, examples in synset_to_examples_map.items():
        bar.update(1)
        bar.set_postfix_str(synset)

        for words, target_idx, cluster, lemma, pos in examples:
            embeddings = embed_words(tokenizer, model, words, target_idx)
            
            # save the embeddings in the dictionary
            if synset not in synset_to_embeddings:
                synset_to_embeddings[synset] = [(embeddings, cluster, lemma, pos)]
            else:
                synset_to_embeddings[synset].append((embeddings, cluster, lemma, pos))
                        
    bar.close()
    
    # save the dictionary in a pickle file
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(synset_to_embeddings, f)

    return synset_to_embeddings


def synset_dist_cosine(synsets_1, synsets_2, threshold=1000, batch_size=100):
    '''
    This function takes as input two lists of embeddings.
    The first one contains a list of embeddings of the synsets of the first synset
    while the second one contains a list of embeddings of the synsets of the second synet.
    The function returns the mean cosine distance between the synsets.

    If the number of synsets is higher than a threshold, then the function calculates the distance in batches.
    '''
    dist = 0
    if len(synsets_1) < threshold and len(synsets_2) < threshold:
        dist_mat = 1 - torch.nn.functional.cosine_similarity(synsets_1[None, :, :], synsets_2[:, None, :], dim=-1)
        dist = dist_mat.mean()
    else:   # calculate the distance in batches.
        num_elements = len(synsets_1) * len(synsets_2)
        sum_of_distances = 0
        batch_size = batch_size
        for i in range(0, len(synsets_1), batch_size):
            dist_mat = 1 - torch.nn.functional.cosine_similarity(synsets_1[None, i:min(i+batch_size, len(synsets_1)), :], synsets_2[:, None, :], dim=-1)
            sum_of_distances += dist_mat.sum()
        
        dist = sum_of_distances / num_elements
    return dist

def synset_dist_euclidean(synsets_1, synsets_2):
    '''
    This function takes as input two lists of embeddings.
    The first one contains a list of embeddings of the synsets of the first synset
    while the second one contains a list of embeddings of the synsets of the second synet.
    The function returns the mean euclidean distance between the synsets.
    '''
    # what cdist does is to calculate the distance between each pair of synsets using the 
    # Lp distance (the same used in Tensor.dist).
    dist = torch.cdist(synsets_1, synsets_2, p=2).mean()
    return dist
    

def cluster_dist(cluster1_synsets, cluster2_synsets, is_intra_cluster=False, p=2, use_pbar=False):
    '''
    This function takes as input two lists.
    The first one contains a list of embeddings of the synsets for the first cluster
    while the second one contains a list of embeddings of the synsets for the second cluster.
    The function returns the mean distance between the synsets.
    '''
    if use_pbar:
        bar = tqdm.tqdm(total=len(cluster1_synsets) * len(cluster2_synsets))
    distances = []
    # for each pair of synsets
    for i in range(len(cluster1_synsets)):
        for j in range(len(cluster2_synsets)):
            if use_pbar:
                bar.update(1)
            if is_intra_cluster and i == j:
                continue
            # calculate the distance between the synsets. Each synset has a variable number of embeddings (one for each example).
            if config.distance_metric == 'cosine':
                dist = synset_dist_cosine(cluster1_synsets[i], cluster2_synsets[j])
            elif config.distance_metric == 'euclidean':
                dist = synset_dist_euclidean(cluster1_synsets[i], cluster2_synsets[j])
            distances.append(dist)
    if use_pbar:
        bar.close()
    return sum(distances) / len(distances)

def print_data_dict(data):
    '''
    Utility to print data.
    '''
    def _print(d):
        if isinstance(d, set):
            print(len(d), end=" ")
        elif isinstance(d, dict):
            print(len(d), end=" ")
        elif isinstance(d, list):
            if len(d) <= 3:
                for el in d:
                    _print(el)
            else:
                print(len(d), end=" ")
        else:
            print(d, end=" ")

    for k, v in data.items():
        print(k, end=": ")
        _print(v)
        print()
    print("\n\n")


def check_number_of_valid_synsets(wn_homonyms, synset_to_examples):
    '''
    Check how many synsets have examples and how many do not have examples.
    If config.check_lemmapos is True, then it also removes from the mapping
    the examples where the lemma and pos is different from the lemmapos.
    '''
    synsets_without_examples = list()
    synsets_with_examples = list()

    bar = tqdm.tqdm(total=len(wn_homonyms))
    for lemmapos, cluster in wn_homonyms.items():
        bar.update(1)
        bar.set_postfix_str(lemmapos)
        for cluster_name, synsets_examples in cluster.items():
            valid_synsets = [s for s in synsets_examples if s[0] in synset_to_examples]
            if len(valid_synsets) > 0:
                synsets_with_examples.extend(valid_synsets)
            non_valid_synsets = [s for s in synsets_examples if s[0] not in synset_to_examples]
            if len(non_valid_synsets) > 0:
                synsets_without_examples.extend(non_valid_synsets)
    
    bar.close()

    return {
        "Number of synsets with examples": len(synsets_with_examples),
        "Number of synsets without examples": len(synsets_without_examples),
        "Number of synsets with examples and no duplicates": len(set(synsets_with_examples)),
        "Number of synsets without examples and no duplicates": len(set(synsets_without_examples))
    }


def get_valid_synsets(synsets, synset_to_embeddings, lemma):
    # get the embeddings of the examples for each synset. Filter embeddings by lemma.
    valid_synsets_embeddings = []
    for synset in synsets:
        valid_embeddings = []
        for embedding, _, l, __ in synset_to_embeddings[synset]:
            if l == lemma:
                valid_embeddings.append(embedding)
        # add all the embeddings to the list
        if len(valid_embeddings) > 0:
            valid_synsets_embeddings.append(torch.stack(valid_embeddings))
        
    return valid_synsets_embeddings

def calculate_inter_cluster_distance(valid_cluster_synsets):

    inter_cluster_list = []
    for i in range(len(valid_cluster_synsets)):
        distances = []
        for j in range(len(valid_cluster_synsets)):
            if i == j:
                continue
            distance = cluster_dist(valid_cluster_synsets[i], valid_cluster_synsets[j])
            distances.append(distance)
        # inter cluster distance for the i-th cluster
        inter_cluster_list.append(float(sum(distances) / len(distances)))
    return inter_cluster_list

def intra_inter_cluster_distance(wn_homonyms, synset_to_embeddings):
    '''
    Calculate the intra cluster distance and all the statistics about the clusters.
    '''

    clusters_with_valid_synsets = dict()


    num_tot_clusters = 0
    num_clusters_with_one_synsets = 0
    num_clusters_with_at_least_two_synsets = 0
    num_clusters_with_one_valid_synsets = 0
    num_clusters_with_at_least_two_valid_synsets = 0
    num_clusters_with_at_least_two_synsets_but_only_one_valid_synset = 0
    num_clusters_with_at_least_two_synsets_but_not_enough_examples_to_calculate_intra = 0
    

    num_lemmapos = len(wn_homonyms)
    num_lemmapos_with_one_cluster = 0
    num_lemmapos_with_one_valid_cluster = 0
    num_lemmapos_with_at_least_two_clusters = 0
    num_lemmapos_with_at_least_two_valid_clusters = 0
    num_lemmapos_with_at_least_two_clusters_but_only_one_valid_cluster = 0
    num_lemmapos_with_at_least_two_clusters_but_not_enough_examples_to_calculate_inter = 0
    num_lemmapos_with_at_least_one_valid_cluster_but_intra_not_possible = 0


    intra_cluster_distances_lemmapos = [] # list that contains the intra cluster distance for each lemmapos
    intra_cluster_distances_clusters = [] # list that contains the intra cluster distance for each cluster

    inter_cluster_distances_lemmapos = []
    inter_cluster_distances_clusters = []

    num_intra_greater_than_inter_lemmapos = 0
    tot_intra_and_inter_lemmapos = 0

    num_intra_greater_than_inter_cluster = 0
    tot_intra_and_inter_cluster = 0

    lemmapos_clusters_mapping = {}

    bar = tqdm.tqdm(total=len(wn_homonyms))

    # Iterate on the lemmapos. For each cluster in the lemmapos calculate the intra cluster distance adn the inter cluster distance.
    for lemmapos, clusters in wn_homonyms.items():
        bar.update(1)
        bar.set_postfix_str(lemmapos)

        lemma = lemmapos[:-2]
        pos = lemmapos[-1]
        lemmapos_clusters_mapping[lemmapos] = {}
        valid_cluster_synsets_embeddings = []    # list of synsets of the cluster
        valid_cluster_ids = []        # list of cluster ids

        intra_cluster_distances = []    # list that contains the intra cluster distance for each cluster in the lemmapos
        # inter cluster distance
        for cluster, synsets_examples in clusters.items():
            num_tot_clusters += 1
            lemmapos_clusters_mapping[lemmapos][cluster] = {}

            # get the list of synsets of the cluster that have embeddings
            synsets_with_embeddings = [s[0] for s in synsets_examples if s[0] in synset_to_embeddings]

            # get the embeddings of the examples for each synset. Filter embeddings by lemma.
            valid_synsets_embeddings = get_valid_synsets(synsets_with_embeddings, synset_to_embeddings, lemma)

            # add the list of synset embeddings to the list of cluster embeddings
            if valid_synsets_embeddings:
                valid_cluster_synsets_embeddings.append(valid_synsets_embeddings)
                valid_cluster_ids.append(cluster)
                lemmapos_clusters_mapping[lemmapos][cluster]["valid"] = True    # the cluster has at least one valid synset
            else:
                lemmapos_clusters_mapping[lemmapos][cluster]["valid"] = False
            
            # add the number of synsets, the number of synsets with embeddings and the number of valid synsets to the dictionary. This will be used to calculate the statistics later.
            num_synsets = len(synsets_examples)
            num_synsets_with_embeddings = len(synsets_with_embeddings)
            num_valid_synsets = len(valid_synsets_embeddings)
            if num_synsets == 1:
                num_clusters_with_one_synsets += 1
            elif num_synsets >= 2:
                num_clusters_with_at_least_two_synsets += 1

            if num_synsets >= 2 and num_valid_synsets == 1:
                num_clusters_with_at_least_two_synsets_but_only_one_valid_synset += 1

            if num_synsets >= 2 and num_valid_synsets <= 1:
                num_clusters_with_at_least_two_synsets_but_not_enough_examples_to_calculate_intra += 1

            lemmapos_clusters_mapping[lemmapos][cluster]["synsets"] = (num_synsets, num_synsets_with_embeddings, num_valid_synsets)
            

            if num_valid_synsets == 0:
                lemmapos_clusters_mapping[lemmapos][cluster]["valid_intra"] = False   # intra cluster distance not possible
            elif num_valid_synsets == 1:
                num_clusters_with_one_valid_synsets += 1
                clusters_with_valid_synsets[(lemma, pos, cluster)] = synsets_examples
                lemmapos_clusters_mapping[lemmapos][cluster]["valid_intra"] = False  # intra cluster distance not possible
            elif num_valid_synsets >= 2:    # at least two valid synsets (intra cluster distance possible)
                num_clusters_with_at_least_two_valid_synsets += 1
                clusters_with_valid_synsets[f"{lemmapos} - {cluster}"] = synsets_examples
                distance = cluster_dist(valid_synsets_embeddings, valid_synsets_embeddings, is_intra_cluster=True)
                intra_cluster_distances_clusters.append(distance)
                intra_cluster_distances.append(distance)
                lemmapos_clusters_mapping[lemmapos][cluster]["valid_intra"] = True  # intra cluster distance possible
                lemmapos_clusters_mapping[lemmapos][cluster]["intra_clsuter_distance"] = distance

        #############################################
        ##### intra cluster distance for the lemmapos

        if len(intra_cluster_distances) > 0:
            intra_cluster_distances_lemmapos.append(float(sum(intra_cluster_distances) / len(intra_cluster_distances)))
            
        num_clusters_in_current_lemmapos = len(clusters)
        num_valid_clusters = len(valid_cluster_synsets_embeddings)

        if num_clusters_in_current_lemmapos == 1:
            num_lemmapos_with_one_cluster += 1
        elif num_clusters_in_current_lemmapos >= 2:
            num_lemmapos_with_at_least_two_clusters += 1


        if num_clusters_in_current_lemmapos >= 2 and num_valid_clusters == 1:
            num_lemmapos_with_at_least_two_clusters_but_only_one_valid_cluster += 1

        if num_clusters_in_current_lemmapos >= 2 and num_valid_clusters <= 1:
            num_lemmapos_with_at_least_two_clusters_but_not_enough_examples_to_calculate_inter += 1

        clusters_where_intra_is_possible = [c for c, v in lemmapos_clusters_mapping[lemmapos].items() if v["valid_intra"]]

        if len(clusters_where_intra_is_possible) == 0 and num_valid_clusters >= 1:
            num_lemmapos_with_at_least_one_valid_cluster_but_intra_not_possible += 1


        lemmapos_clusters_mapping[lemmapos]["num_valid_clusters"] = num_valid_clusters


        if num_valid_clusters == 0:
            lemmapos_clusters_mapping[lemmapos]["valid_inter"] = False
        elif num_valid_clusters == 1:
            num_lemmapos_with_one_valid_cluster += 1
            lemmapos_clusters_mapping[lemmapos]["valid_inter"] = False
            
        elif num_valid_clusters >= 2:   # inter cluster distance possible
            num_lemmapos_with_at_least_two_valid_clusters += 1

            lemmapos_clusters_mapping[lemmapos]["valid_inter"] = True

            # calculate the inter cluster distance
            inter_cluster_list = calculate_inter_cluster_distance(valid_cluster_synsets_embeddings)
            # inter cluster distance for the lemmapos
            inter_cluster_distance_lemmapos = float(sum(inter_cluster_list) / len(inter_cluster_list))
            lemmapos_clusters_mapping[lemmapos]["inter_cluster_distance"] = inter_cluster_distance_lemmapos
            inter_cluster_distances_lemmapos.append(inter_cluster_distance_lemmapos)

            # inter cluster distance for each cluster
            for i in range(len(valid_cluster_ids)):
                lemmapos_clusters_mapping[lemmapos][valid_cluster_ids[i]]["inter_cluster_distance"] = inter_cluster_list[i]
            inter_cluster_distances_clusters.extend(inter_cluster_list)

            # check how many times the inter cluster distance is lower than the intra cluster distance
            if len(intra_cluster_distances) > 0:
                tot_intra_and_inter_lemmapos += 1
                if inter_cluster_distance_lemmapos < intra_cluster_distances_lemmapos[-1]:  # I can use -1 because the last element is the one I just added
                    num_intra_greater_than_inter_lemmapos += 1
                for el in intra_cluster_distances:
                    tot_intra_and_inter_cluster += 1
                    if inter_cluster_distance_lemmapos < el:
                        num_intra_greater_than_inter_cluster += 1
    
    bar.close()

    # calculate the statistics about the clusters

    stats = {
        "all_clusters": num_tot_clusters,
        "clusters_with_one_synsets": num_clusters_with_one_synsets,
        "clusters_with_at_least_two_synsets": num_clusters_with_at_least_two_synsets,
        "clusters_with_one_valid_synsets":  num_clusters_with_one_valid_synsets,
        "clusters_with_at_least_two_valid_synsets": num_clusters_with_at_least_two_valid_synsets,
        "clusters_with_at_least_two_synsets_but_only_one_valid_synset": num_clusters_with_at_least_two_synsets_but_only_one_valid_synset,
        "clusters_with_at_least_two_synsets_but_not_enough_examples": num_clusters_with_at_least_two_synsets_but_not_enough_examples_to_calculate_intra,
        "intra_cluster_distances_lemmapos": [intra_cluster_distances_lemmapos, f"Mean: {sum(intra_cluster_distances_lemmapos) / len(intra_cluster_distances_lemmapos)}"],
        "intra_cluster_distances_clusters": [intra_cluster_distances_clusters, f"Mean: {sum(intra_cluster_distances_clusters) / len(intra_cluster_distances_clusters)}"],
        "all_lemmapos": num_lemmapos,
        "lemmapos_with_one_cluster": num_lemmapos_with_one_cluster,
        "lemmapos_with_one_valid_cluster": num_lemmapos_with_one_valid_cluster,
        "lemmapos_with_at_least_two_clusters": num_lemmapos_with_at_least_two_clusters,
        "lemmapos_with_at_least_two_valid_clusters": num_lemmapos_with_at_least_two_valid_clusters,
        "lemmapos_with_at_least_two_clusters_but_only_one_valid_cluster": num_lemmapos_with_at_least_two_clusters_but_only_one_valid_cluster,
        "lemmapos_with_at_least_two_clusters_but_not_enough_examples": num_lemmapos_with_at_least_two_clusters_but_not_enough_examples_to_calculate_inter,
        "lemmapos_with_at_least_one_valid_cluster_but_intra_not_possible": num_lemmapos_with_at_least_one_valid_cluster_but_intra_not_possible,
        "inter_cluster_distances_lemmapos": [inter_cluster_distances_lemmapos, f"Mean: {sum(inter_cluster_distances_lemmapos) / len(inter_cluster_distances_lemmapos)}"],
        "inter_cluster_distances_clusters": [inter_cluster_distances_clusters, f"Mean: {sum(inter_cluster_distances_clusters) / len(inter_cluster_distances_clusters)}"],
        "num_intra_greater_than_inter_lemmapos": [num_intra_greater_than_inter_lemmapos, f"{num_intra_greater_than_inter_lemmapos} / {tot_intra_and_inter_lemmapos}", f"{num_intra_greater_than_inter_lemmapos / tot_intra_and_inter_lemmapos}"],
        "num_intra_greater_than_inter_cluster": [num_intra_greater_than_inter_cluster, f"{num_intra_greater_than_inter_cluster} / {tot_intra_and_inter_cluster}", f"{num_intra_greater_than_inter_cluster / tot_intra_and_inter_cluster}"]
    }
    
    return stats, clusters_with_valid_synsets, lemmapos_clusters_mapping


def calculate_mean_cluster_distance(all_valid_clusters, synset_to_embeddings, num_samples=1000, seed=42):
    '''
    Calculate the mean distance between clusters.
    This function samples "num_samples" clusters and calculates the distance between them.
    '''
    all_cluster_synset_embeddings = []
    # set seeds
    random.seed(seed)
    # sample random and calculate the distance
    valid_cluster_random_sample = random.sample(list(all_valid_clusters.items()), num_samples)
    for cluster, synsets_examples in valid_cluster_random_sample:
        valid_synsets_embeddings = torch.cat([torch.stack([item[0] for item in synset_to_embeddings[s[0]]])  for s in synsets_examples if s[0] in synset_to_embeddings])
        all_cluster_synset_embeddings.append(valid_synsets_embeddings)
    print(len(all_cluster_synset_embeddings))
    mean_distance = cluster_dist(all_cluster_synset_embeddings, 
                                all_cluster_synset_embeddings, 
                                is_intra_cluster=True,
                                use_pbar=True)  # note, even if it is not intra, 
                                                        # I am still passing the same list twice, 
                                                        # so I want to skip the diagonal values where 
                                                        # elements are compared with them selves.
    return mean_distance


def generate_venn():
    '''
    This function opens the full resource and generates a venn diagram that shows the overlap between
    the synsets of the different datasets.
    '''
    sets = []
    for path in config.dataset_paths:
        print("Loading", path)
        dataset = open_dataset(path)
        s = map_synset_to_examples(dataset)
        sets.append(set(s.keys()))
    with open(config.wn_examples_path, 'r') as f:
        wn_dataset = [json.loads(line) for line in f]
    s = map_synset_to_wn_examples(wn_dataset)
    sets.append(set(s.keys()))
    from venny4py.venny4py import venny4py
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    data = {
        "SemCor": sets[0],
        "ALL": sets[1],
        "SemEval": sets[2],
        "WordNet": sets[3]
    }
    venny4py(sets=data, out="images/venn.png")
    plt.cla()


def process_dataset(wn_homonyms, synset_to_embeddings):
    '''
    This function calculates the statistics about the clusters and the lemmapos.
    It also plots the histograms and the density functions.
    '''

    intra_inter_cluster_data, clusters_with_valid_synsets, _ = intra_inter_cluster_distance(wn_homonyms, synset_to_embeddings)
    print_data_dict(intra_inter_cluster_data)


    mean_cluster_distance = calculate_mean_cluster_distance(clusters_with_valid_synsets, synset_to_embeddings, num_samples=1000, seed=42)
    print("Mean Inter Cluster Cohesion for all clusters:", -mean_cluster_distance)


    ############################
    # PLOT THE HISTOGRAMS

    intra_cluster_list = -torch.asarray(intra_inter_cluster_data["intra_cluster_distances_lemmapos"][0])
    inter_cluster_list = -torch.asarray(intra_inter_cluster_data["inter_cluster_distances_lemmapos"][0])

    intra_cluster_cohesion_list = -torch.asarray(intra_cluster_list)
    inter_cluster_cohesion_list = -torch.asarray(inter_cluster_list)
    # plot the intra cluster distance vs the inter cluster distance in an histogram
    plt.hist(intra_cluster_cohesion_list, bins=100, alpha=0.5, label='Average Cluster Cohesion')
    plt.hist(inter_cluster_cohesion_list, bins=100, alpha=0.5, label='(Lemma, POS) Cohesion')
    plt.legend(loc='upper right')
    plt.xlabel('Cohesion')
    plt.ylabel('Density')
    plt.savefig(f'images/intra_inter_cluster_distance_hist-{config.model_name.split("/")[-1]}-{config.distance_metric}-.pdf', format='pdf')
    plt.clf()



    # now plot the density function as a smooth line

    intra_cluster_list_greater_than_zero = intra_cluster_list[intra_cluster_list > 0]
    inter_cluster_list_greater_than_zero = inter_cluster_list[inter_cluster_list > 0]
    print("Number of lemmapos where the intra cluster distance is greater than zero:", len(intra_cluster_list_greater_than_zero))
    print("Number of lemmapos where the inter cluster distance is greater than zero:", len(inter_cluster_list_greater_than_zero))

    sns.kdeplot(x=intra_cluster_list, fill=True, label='Average Cluster Cohesion', bw_method="scott")
    sns.kdeplot(x=inter_cluster_list, fill=True, label='(Lemma, POS) Cohesion', bw_method="scott")
    plt.legend(loc='upper right')
    plt.xlabel('Cohesion')
    plt.ylabel('Density')
    plt.savefig(f"images/intra_inter_cluster_distance_smooth-{config.model_name.split('/')[-1]}-{config.distance_metric}-.pdf", format="pdf")
    plt.clf()



def add_inter_and_intra_cluster_distances_to_fails(fails, coarse2fine, synset_to_embeddings):
    '''
    for each entry in the fails dictionary, iterate on the clusters, 
    retrieve the embeddings of the synsets and calculate the distance between them.
    '''
    correct_pred_and_intra_greater_than_inter = 0
    correct_pred_and_intra_lower_than_inter = 0
    wrong_pred_and_intra_greater_than_inter = 0
    wrong_pred_and_intra_lower_than_inter = 0

    for _, data in fails.items():
        if not isinstance(data, dict):
            continue
        instance = data["instance"]
        key = list(instance["candidate_clusters"].keys())[0]
        candidate_clusters = instance["candidate_clusters"][key]
        lemma = instance["lemma"][key]

        # get the synsets of the cluster.
        # for each cluster, get all the synsets that have embeddings.
        # then, for each synset, get the embeddings of all the examples that have 
        # the same lemma as the instance.

        cluster_synsets = []
        for cluster in candidate_clusters:
            # get the synsets of the cluster
            synsets = []
            for entry in coarse2fine[cluster]:
                synsets.append(entry[0])
            # filter the synsets that have embeddings
            synsets = [s for s in synsets if s in synset_to_embeddings]

            # get the embeddings of the examples for each synset. Filter embeddings by lemma.
            synset_embeddings = []
            for synset in synsets:
                valid_embeddings = []
                for embedding, c, l, p in synset_to_embeddings[synset]:
                    if (l == lemma and config.check_lemmapos) or (not config.check_lemmapos):
                        valid_embeddings.append(embedding)
                # add all the embeddings of the examples of the synset to the list
                if len(valid_embeddings) > 0:
                    synset_embeddings.append(torch.stack(valid_embeddings))
            if len(synset_embeddings) == 0:
                print("No valid embeddings for cluster", cluster)
            # add the list of synset embeddings to the list of cluster embeddings
            cluster_synsets.append(synset_embeddings)
        
        # calculate the intra cluster distance
        intra_cluster_list = []
        for synset_embeddings in cluster_synsets:
            if len(synset_embeddings) == 0:
                intra_cluster_list.append(-1)
                continue
            if len(synset_embeddings) == 1:
                intra_cluster_list.append(0)
                continue
            distance = cluster_dist(synset_embeddings, synset_embeddings, is_intra_cluster=True)
            intra_cluster_list.append(float(distance.item()))
        
        # calculate the inter cluster distance
        inter_cluster_list = []
        for i in range(len(cluster_synsets)):
            distances = []
            for j in range(len(cluster_synsets)):
                if i == j:
                    continue
                distance = cluster_dist(cluster_synsets[i], cluster_synsets[j])
                distances.append(distance)
            inter_cluster_list.append(float(sum(distances) / len(distances)))
        
        max_intra = max(intra_cluster_list)
        min_inter = min(inter_cluster_list)
        if max_intra > min_inter:
            if data["correct"]:
                correct_pred_and_intra_greater_than_inter += 1
            else:
                wrong_pred_and_intra_greater_than_inter += 1
        else:
            if data["correct"]:
                correct_pred_and_intra_lower_than_inter += 1
            else:
                wrong_pred_and_intra_lower_than_inter += 1
        
        # add the distances to the dictionary
        data["intra_cluster_distance"] = intra_cluster_list
        data["inter_cluster_distance"] = inter_cluster_list

    print("Correct and intra greater than inter:", correct_pred_and_intra_greater_than_inter)
    print("Correct and intra lower than inter:", correct_pred_and_intra_lower_than_inter)
    print("Wrong and intra greater than inter:", wrong_pred_and_intra_greater_than_inter)
    print("Wrong and intra lower than inter:", wrong_pred_and_intra_lower_than_inter)

    return fails



def read_dataset_from_new_split():
    '''
    Read data from the train-test-dev split and create a
    mapping from synset to examples.
    '''
    
    def merge_mapping(dict_1, dict_2):
        for key, value in dict_2.items():
            if key in dict_1:
                for idx in range(len(value["example_tokens"])):
                    entry = (value["example_tokens"][idx], 
                             value["instance_ids"][idx], 
                             value["cluster_name"], 
                             value["lemma"][idx], 
                             value["pos"][idx])
                    dict_1[key].append(entry)
            else:
                l = [] 
                for idx in range(len(value["example_tokens"])):
                    entry = (value["example_tokens"][idx], 
                             value["instance_ids"][idx], 
                             value["cluster_name"], 
                             value["lemma"][idx], 
                             value["pos"][idx])
                    l.append(entry)

                dict_1[key] = l
        return dict_1
    
    mapping = {}

    for path in config.dataset_paths:
        print(path)
        mapping = merge_mapping(mapping, read_examples_mapping(path))
    print(len(list(mapping.keys())))
    
    return mapping


def load_from_full_reource():
    '''
    Read data from the full resource and create a
    mapping from synset to examples.
    '''
    synset_to_examples_map = {}
    for path in config.dataset_paths:
        print("Loading:", path)
        dataset = open_dataset(path)
        synset_to_examples_map = map_synset_to_examples(dataset, synset_to_examples_map)
        print(len(synset_to_examples_map))
            
    with open(config.wn_examples_path, 'r') as f:
        wn_dataset = [json.loads(line) for line in f]
    synset_to_examples_map = map_synset_to_wn_examples(wn_dataset, synset_to_examples_map)
    print(len(synset_to_examples_map))

    del wn_dataset, dataset

    return synset_to_examples_map

