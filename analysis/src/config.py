
# model

model_name = 'bert-large-cased' # 'bert-large-cased'

models = ['bert-large-cased', 'microsoft/deberta-v3-large']

device = 'cuda'

# True if you want to calculate the embeddings for the examples in the dataset
# otherwise it will load the embeddings from file
EMBED_EXAMPLES = False

# experiment variables
distance_metric = 'cosine'  # 'euclidean' or 'cosine


distance_metric_list = ['cosine', 'euclidean']

# distance_metric_factor = 2 if distance_metric == 'cosine' else 1  # set to 2 for cosine distance, 1 for euclidean distance

check_lemmapos = True  # True if you want to check the lemma and pos of the examples to be the same as the id of the lemma.pos

universal_pos2wn_pos = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
wn_pos2universal_pos = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}

universal_and_wn_to_wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r", "n":"n", "v":"v", "a":"a", "r":"r"} # looks dumb but allows to generalize later without need for try except clauses or similar approaches  

# dataset paths

data_base_path = './final_data/'

cluster_path = f'{data_base_path}/wn_homonyms.pkl'   # path to the cluster of homonyms

coarse2fine_path = f'{data_base_path}/cluster2fine_map.json'

# paths to the datasets
use_train_test_dev = True

if use_train_test_dev:
    dataset_paths = [f"{data_base_path}/new_split/train.json",
            f"{data_base_path}/new_split/dev.json",
            f"{data_base_path}/new_split/test.json"]
else:
    dataset_paths = [f'{data_base_path}/wsd_datasets/training_sets/mapped_semcor.json',
                        f'{data_base_path}/wsd_datasets/evaluation_sets/mapped_ALLamended.json',
                        f'{data_base_path}/wsd_datasets/evaluation_sets/mapped_semeval2007.json']

wn_examples_path = f'{data_base_path}/wsd_datasets/training_sets/mapped_wn_examples.jsonl'   # path to the wordnet examples


embeddings_path = './embeddings/'  # path to save the embeddings