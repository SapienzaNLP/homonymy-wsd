<div align="center">

# Analyzing Homonymy Disambiguation Capabilities of Pretrained Language Models

[![Conference](https://img.shields.io/badge/LREC--COLING-2024-blue
)](https://lrec-coling-2024.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2024.lrec-main.83/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

</div>

## About the Project

This project investigates the ability of Pretrained Language Models (PLMs) to perform **Homonymy Disambiguation**, a specific form of Word Sense Disambiguation (WSD) focused on distinguishing **unrelated senses** of the same word—i.e., homonyms. To this end, we introduce a **new large-scale annotated resource** that reduces the granularity of [WordNet](https://wordnet.princeton.edu/) by systematically separating homonymous (i.e., unrelated) meanings and grouping polysemous (i.e., related) ones.

The repository provides:
- the new resource—consisting of a homonymy-based clustering of WordNet senses—together with all the data required to replicate the experiments presented in the paper;
- tools to **probe PLMs** and investigate whether the ability to distinguish homonymous senses emerges during pretraining—without task-specific fine-tuning—using **distance-based measures**;
- scripts to **fine-tune and evaluate models** on both **coarse-grained** and **fine-grained** WSD tasks.
- the code to **analyze** the new homonymy-based resource.

Our experiments show that PLMs such as BERT can separate homonymous senses with **up to 95% accuracy** without any fine-tuning, and that a simple fine-tuned model can reach even higher performance.

## Structure of the Repository

To run the code, first download the required [data file](https://drive.google.com/file/d/1kKQzOpTgvfCFUDs9H_d0z1PkcB3yez3q/view?usp=sharing) and unzip it. Then, create a new [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment and install the required packages using the following commands:

```bash
conda create -n homonymy_wsd python=3.9
conda activate homonymy_wsd

pip install -r requirements.txt
```

The code is organized as follows:
- **data**: new resource and data required to replicate the experiments from the paper;
- **distances_analysis**: code to probe the ability of PLMs to disambiguate homonyms. This is done employing distance measures between contextualized embeddings (instance-based learning);
- **wsd_systems**: code to train and evaluate PLMs on the fine-grained and coarse-grained WSD tasks;
- **resource_analysis**: code to analyze the new resource described in the paper, computing intra-cluster and inter-cluster statistics.

## Data

The *data* folder includes both the new resource and all the data required to replicate the experiments presented in the paper.

### Resource

The new resource is stored in the *data/wn_homonyms.pkl* and *data/cluster2fine_map.json* files:

- **wn_homonyms.pkl**: a Python dictionary where each key is a WordNet (lemma, PoS) pair, and each value contains the homonymy-based clustering of its possible WordNet senses (or meanings);
- **cluster2fine_map.json**: a mapping from homonymy cluster identifiers to lists of WordNet synsets (with definitions) that form the clusters.

For example, the following Python code snippet extracts the homonymy-based clustering of the WordNet senses for the **(list, VERB)** pair, as shown in Table 1 of the paper:

```python
import pickle                                                                                                                                                                                               

with open('data/wn_homonyms.pkl', 'rb') as handle: 
    lemma_pos2homonymy_clusters = pickle.load(handle)

>>> lemma_pos2homonymy_clusters['list.v']
{
    'list.v.h.01': [('list.v.01', 'give or make a list of; name individually; give the names of'),
                    ('list.v.02', 'include in a list'),
                    ('number.v.03', 'enumerate')],
    'list.v.h.02': [('list.v.03', 'cause to lean to the side'),
                    ('list.v.04', 'tilt to one side')]
 }
```

To retrieve the WordNet synset identifiers (and their definitions) contained in these homonymy clusters, you can use the following code:

```python
import json

with open('data/cluster2fine_map.json', 'r', encoding='utf-8') as f: 
    homonymy_cluster_id2wn_synsets = json.load(f)

>>> homonymy_cluster_id2wn_synsets['list.v.h.01']
[
    ['list.v.01', 'give or make a list of; name individually; give the names of'],
    ['list.v.02', 'include in a list'],
    ['number.v.03', 'enumerate']
]

>>> homonymy_cluster_id2wn_synsets['list.v.h.02']
[
    ['list.v.03', 'cause to lean to the side'],
    ['list.v.04', 'tilt to one side']
]
```

### Mapped WSD Datasets

The *train/dev/test* split used in the experiments is located in the *data/new_split* folder.  
Each `.json` file in this folder is a dictionary where each key uniquely identifies a sense-tagged sentence, and each value is a dictionary containing both fine-grained (WordNet senses) and coarse-grained (homonymy clusters) annotations.

The sense-tagged sentences fall into two categories:

1. **WordNet Examples**: usage examples from WordNet, each containing exactly one tagged WordNet sense;
2. **SemCor, SemEval-2007, and ALL<sub>NEW</sub>**: annotated corpora containing multiple sense-tagged words per sentence.

#### Field Descriptions

<details>
<summary><strong>Type 1 – WordNet Examples</strong></summary>

| Field               | Description                                                    |
|---------------------|----------------------------------------------------------------|
| `synset_name`       | Gold WordNet synset ID of the tagged (lemma, PoS) instance     |
| `pos`               | Part-of-speech tag of the tagged sense                         |
| `offset`            | WordNet offset of the tagged sense                             |
| `lemma`             | Lemma of the tagged sense                                      |
| `wn_candidates`     | List of candidate WordNet senses for the (lemma, PoS) instance |
| `cluster_name`      | Gold homonymy cluster ID for the tagged instance               |
| `candidate_clusters`| List of candidate homonymy clusters for the tagged instance    |
| `example_tokens`    | Sentence words                                                 |
| `instance_ids`      | Word indexes of the tagged (lemma, PoS) instance               |

</details>

<details>
<summary><strong>Type 2 – SemCor / SemEval-2007 / ALL<sub>NEW</sub></strong></summary>

| Field                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `instance_ids`       | Dictionary mapping word indexes to tagged (lemma, PoS) instance identifiers |
| `lemmas`             | Lemmas of the sentence                                                      |
| `pos_tags`           | Part-of-speech tags of the sentence                                         |
| `senses`             | Dictionary mapping word indexes to their gold WordNet synset IDs            |
| `words`              | Sentence words                                                              |
| `wn_candidates`      | Dictionary mapping word indexes to candidate WordNet senses                 |
| `gold_clusters`      | Dictionary mapping word indexes to their gold homonymy cluster IDs          |
| `candidate_clusters` | Dictionary mapping word indexes to candidate homonymy cluster IDs           |

</details>

## Probing PLMs

To probe PLMs for the Homonymy Disambiguation (HD) task using distance measures, run the following command:

```bash
python distances_analysis/src/main_distances_analysis.py > logs/distances_analysis.txt
```

Afterwards, the output file *distances_analysis.txt* will contain the results of the probing experiments reported in the paper.


## WSD Systems Comparison

We fine-tuned the **BERT** pre-trained language model with a token classification training. *WandB* training logs and model checkpoints can be found at the following links:

> **WandB trainings**: https://wandb.ai/lavallone/homonyms?nw=nwuserlavallone <br> **Model checkpoints**: https://drive.google.com/drive/folders/1f2vJpmgvTIAGFLEioWJjiyDvowxTQUNI?usp=sharing

To start the fine-tuning, run:

```
python wsd_systems/main_wsd_plm.py --mode train --type coarse/fine --encoder bert
```

---------------------------------------------------------------------------------------
We evaluated the fine-tuned models on the ***fine-grained*** and ***coarse-grained*** WSD tasks. For the *coarse-grained* task, in addition to the *COARSE* model, we also employed the *FINE* one to predict homonymy clusters (*FINE_2_CLUSTER*).

The evaluation has been carried out on three different portions of the test set mentioned in the paper:
1. the entire *test* set (*test.json*);
2. only instances with at least two candidate clusters;
3. the subset of the *test* set employed for the probing experiments (*test_dictionary_lemma.pkl*).

1️⃣ To evaluate on the entire *test* set using COARSE, FINE, and FINE_2_CLUSTER, run:
```
python wsd_systems/main_wsd_plm.py --mode eval --type coarse/fine/fine2cluster --encoder bert --model path_model
```

2️⃣ To evaluate only on instances with at least two candidate clusters, you need to switch on the *cluster_candidates_filter* hyperparameter defined in the *src/hyperparameters.py* file and run the command reported above.

3️⃣ To evaluate on the subset of the test set defined in *test_dictionary_lemma.pkl* (COARSE and FINE), run:
```
python wsd_systems/main_wsd_plm.py --mode eval --type base_subset --encoder bert --model path_model
```
whereas for FINE_2_CLUSTER:
```
python wsd_systems/main_wsd_plm.py --mode eval --type fine2cluster_subset --encoder bert --model path_model
```


## Resource Analysis:
For the analysis of the new resource described in the paper, run the following command:

```bash
python resource_analysis/src/main_analyze_resources.py > logs/resource_analysis.txt
```

It will compute statistics about intra-cluster and inter-cluster distances.
Images will be saved in the *images* folder, while the statistics will be saved in the *logs* folder.

## Cite this Work
If you use any part of this work, please consider citing the paper as follows:

```bibtex
@inproceedings{proietti-etal-2024-analyzing-homonymy,
    title = "Analyzing Homonymy Disambiguation Capabilities of Pretrained Language Models",
    author = "Proietti, Lorenzo  and
      Perrella, Stefano  and
      Tedeschi, Simone  and
      Vulpis, Giulia  and
      Lavalle, Leonardo  and
      Sanchietti, Andrea  and
      Ferrari, Andrea  and
      Navigli, Roberto",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.83",
    pages = "924--938",
}
```

## License
This work is licensed under [Creative Commons Attribution-ShareAlike-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
