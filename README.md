<div align="center">

# Analyzing Homonymy Disambiguation Capabilities of Pretrained Language Models

[![Conference](https://img.shields.io/badge/LREC--COLING-2024-blue
)](https://lrec-coling-2024.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2024.lrec-main.83/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

</div>

## About the Project

This is the official repository of the paper [*Analyzing Homonymy Disambiguation Capabilities of Pretrained Language Models*](https://aclanthology.org/2024.lrec-main.83/).

## Structure of the Repository

The code is organized as follows:
- **analysis**: contains the code to analyze the new resource described in the paper, computing intra-cluster and inter-cluster statistics;
- **distance_analysis**: contains the code to evaluate the capability of Pretrained Language Models (PLMs) to disambiguate homonyms. This is done employing distance measures between contextualized embeddings (instance-based learning);
- **probing_LM**: contains the code to train and evaluate PLMs on the fine-grained and coarse-grained Word Sense Disambiguation (WSD) tasks.

To run the code, unzip the *_final_data.zip* file, create a new *conda environment*, and then install the *requirements* using the following commands:

```bash
unzip _final_data.zip

conda create -n homonymy_wsd python=3.9
conda activate homonymy_wsd

pip install -r requirements.txt
```

## Probing PLMs

To probe PLMs for the Homonymy Disambiguation (HD) task using distance measures, run the following command:

```bash
python distances_analysis/main_distances_analysis.py > distances_analysis.txt
```

Afterwards, the output file *distances_analysis.txt* will contain the results of the probing experiments reported in the paper.


## Disambiguation Systems Comparison

We fine-tuned the **BERT** pre-trained language model with a token classification objective. *WandB* training logs and model checkpoints can be found at the following links:

> **WandB trainings**: https://wandb.ai/lavallone/homonyms?nw=nwuserlavallone <br> **Model checkpoints**: https://drive.google.com/drive/folders/1f2vJpmgvTIAGFLEioWJjiyDvowxTQUNI?usp=sharing

To start the fine-tuning, run:

```
python probing_LM/main_probing_LM.py --mode train --type coarse/fine --encoder bert
```

---------------------------------------------------------------------------------------
We evaluated the fine-tuned models on the ***fine-grained*** and ***coarse-grained*** WSD tasks. For the *coarse-grained* task, in addition to the *COARSE* model, we also employed the *FINE* one to predict homonymy clusters (*FINE_2_CLUSTER*).

The evaluation has been carried out on three different portions of the test set mentioned in the paper:
1. the entire *test* set (*test.json*);
2. only instances with at least two candidate clusters;
3. the subset of the *test* set employed for the probing experiments (*test_dictionary_lemma.pkl*).

1️⃣ To evaluate on the entire *test* set using COARSE, FINE, and FINE_2_CLUSTER, run:
```
python probing_LM/main_probing_LM.py --mode eval --type coarse/fine/fine2cluster --encoder bert --model path_model
```

2️⃣ To evaluate only on instances with at least two candidate clusters, you simply need to switch on the *cluster_candidates_filter* hyperparameter defined in the *src/hyperparameters.py* file, and run the command defined above.

3️⃣ To evaluate on the subset of the test set defined in *test_dictionary_lemma.pkl* (COARSE and FINE), run:
```
python probing_LM/main_probing_LM.py --mode eval --type base_subset --encoder bert --model path_model
```
whereas for FINE_2_CLUSTER:
```
python probing_LM/main_probing_LM.py --mode eval --type fine2cluster_subset --encoder bert --model path_model
```


## Analyses:
For the analyses of the new resource described in the paper, run the following command:

```bash
python src/analysis/main_analyze_resources.py > logs/analyses.txt
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
This work is licensed under the CC BY-NC-SA 4.0 license.