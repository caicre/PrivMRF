# Data Synthesis via Differentially Private Markov Random Fields

Kuntai Cai, Xiaoyu Lei, Jianxin Wei, Xiaokui Xiao

caikt@comp.nus.edu.sg

This project provides the implementation of the paper "Data Synthesis via Differentially Private Markov Random Fields" (PrivMRF). It enables generating a synthetic dataset with differential privacy guarantees. 

## 1. Setup

Requirements:
- Python 3.10 or later
- Optional: an NVIDIA GPU with CUDA 12 drivers. PrivMRF uses the GPU through `cupy` when it is available, and runs on CPU otherwise.

Install dependencies with [uv](https://docs.astral.sh/uv/):

```
uv sync                # CPU only
uv sync --extra gpu    # with GPU support (cupy)
```

Or install the dependencies listed in `pyproject.toml` with pip, e.g. `pip install numpy scipy pandas networkx matplotlib scikit-learn` (add `"cupy-cuda12x[ctk]"` for GPU support).

## 2. Usage

### 2.1 Reproduce Paper Results

To reproduce the experimental results from the paper:

```
uv run python script.py
```

This will run PrivMRF once on each of the four datasets and six ε values, reporting the total variation distances (TVD). Runtime is 2-5 hours depending on hardware.

You may modify `script.py` to run a subset of the experiments.

### 2.2 Generate Synthetic Data

To generate a synthetic dataset with specified settings:

```
uv run python main.py
```

This generates synthetic data without reporting metrics.

### 2.3 SVM Experiments

SVM experiment code in `script.py` is commented out by default. Uncomment to reproduce:

```python
run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25) 
run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25, generate=False)
```

The first line generates synthetic data, the second trains and tests SVMs in parallel. Limit `data_list`, `method_list`, `epsilon_list` sizes to control number of processes. One dataset/ε takes 1-6 hours.

## 3. Custom Datasets 

`main.py` shows how to run PrivMRF with default config. Read your data, preprocess domains to discrete int values, and call `PrivMRF` to train and generate synthetic data.

PrivMRF is designed for discrete attributes, so continuous attributes need to be discretized (e.g., binned) first.

Settings are passed as `p_config` to `PrivMRF.run()`; see `DEFAULT_CONFIG` in `PrivMRF/main.py` for all of them. For example, δ is set by `config['delta']` (default 1e-5), and `config['gpu']` chooses CPU or GPU (default: GPU if available).

`preprocess()` in `PrivMRF/preprocess.py` maps the values of a dataset in `./data` to int values, and `postprocess()` maps synthetic data back to the original values.

### 3.1 Attribute Hierarchy

In the `adult` dataset, for each attribute, we merge values into bins. Since the number of data records in a bin is larger than the number of records of each value, merging values provides more resistance to noise. This slightly improves PrivMRF performance on `adult` but doesn't always help for other datasets.

To use attribute hierarchy for your dataset:
1. Define hierarchy (see `data/adult_hierarchy.json`) 
2. Read with `read_hierarchy()` in `PrivMRF/attribute_hierarchy.py`
3. Pass hierarchy to `PrivMRF.run()`  
4. Set `config['enable_attribute_hierarchy'] = True`

## 4. Privacy Budget

PrivMRF computes the privacy budget of the Gaussian mechanism for any ε and δ with `privacy_budget()` in `PrivMRF/utils/tools.py` (Lemma 3 of the paper).
The budget is allocated as in Section 3.6 of the paper: 10% for constructing the attribute graph, 1% for the number of records, 10% for selecting marginals, and 79% for marginal distributions.

## Code Reference 

https://github.com/ryan112358/private-pgm