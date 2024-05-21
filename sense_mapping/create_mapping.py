import json

if __name__ == '__main__':
    # COARSE MAPPING
    d = json.load(open("cluster2fine_map.json", "r"))
    all_senses_list = list(d.keys())
    sense2id = {}
    id2sense = {}
    idx=0
    for sense in all_senses_list:
        sense2id[sense] = idx
        id2sense[idx] = sense
        idx+=1
    sense2id["<UNK>"] = idx
    id2sense[idx] = "<UNK>"
    json.dump(sense2id, open("cluster_sense2id.json", "w"))
    json.dump(id2sense, open("cluster_id2sense.json", "w"))
 
	# FINE MAPPING
    all_senses_list = []
    for k in d.keys():
        for fine_s in d[k]:
            all_senses_list.append(fine_s[0])
    all_senses_list = list(set(all_senses_list))
    sense2id = {}
    id2sense = {}
    idx=0
    for sense in all_senses_list:
        sense2id[sense] = idx
        id2sense[idx] = sense
        idx+=1
    sense2id["<UNK>"] = idx
    id2sense[idx] = "<UNK>"	
    json.dump(sense2id, open("fine_sense2id.json", "w"))
    json.dump(id2sense, open("fine_id2sense.json", "w"))