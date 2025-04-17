from embeddings_extractor import *
import config
import pickle
import os


if __name__ == '__main__':

    # # create a map from  synset to examples for that synset
    # # by reading the dataset
    if config.use_train_test_dev:
        synset_to_examples_map = read_dataset_from_new_split()
    else:
        synset_to_examples_map = load_from_full_reource() 

    # check how many synsets have examples and how many don't

    # open the cluster mapping
    with open(config.cluster_path, 'rb') as f:
        wn_homonyms = pickle.load(f)

    ret = check_number_of_valid_synsets(wn_homonyms, synset_to_examples_map)
    print_data_dict(ret)

    distance_metric_list = config.distance_metric_list

    # for each model and distance metric, we will embed the examples and calculate the statistics
    for model_name in config.models:
        config.model_name = model_name
        for distance_metric in distance_metric_list:
            config.distance_metric = distance_metric

            # generate a venn diagram that shows the aynset overlap between the datasets
            # generate_venn()

            print("=====================================================")
            print("Model:", config.model_name)
            print("Distance Metric:", config.distance_metric)
            print()

            if not os.path.exists(config.embeddings_path):
                os.makedirs(config.embeddings_path)

            save_path = os.path.join(config.embeddings_path, f"{config.model_name.split('/')[-1]}-using-train-dev-test-{str(config.use_train_test_dev)}.pkl")
            # embed the examples with the model    
            if config.EMBED_EXAMPLES and distance_metric == distance_metric_list[0]:    # if we want to embed the examples and it is the first distance metric, 
                                                                                        # then we embed the examples. Otherwise we load from file.
                tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
                model = AutoModel.from_pretrained(config.model_name)
                model.eval()
                model.to(config.device)
                synset_to_embeddings = map_synsets_to_embeddings(tokenizer, model, synset_to_examples_map, save_path = save_path)

            else:
                # load the embeddings from the pickle file
                with open(save_path, 'rb') as f:
                    synset_to_embeddings = pickle.load(f)

            # calculate statistics about the clusters and the lemmapos
            process_dataset(wn_homonyms, synset_to_embeddings)
