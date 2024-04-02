# Data Synthesis via Differentially Private Markov Random Fields

Kuntai Cai, Xiaoyu Lei, Jianxin Wei, Xiaokui Xiao

caikt@comp.nus.edu.sg

This project provides the implementation of the paper "Data Synthesis via Differentially Private Markov Random Fields" (PrivMRF). It enables generating a synthetic dataset with differential privacy guarantees. 

## 1. Setup

Requirements:
- Python 3.8 
- CUDA 11.7
- GPU that supports `cupy`

Install dependencies:

```
pip3 install -r requirements.txt
```

## 2. Usage

### 2.1 Reproduce Paper Results

To reproduce the experimental results from the paper:

```
python3 script.py
```

This will run PrivMRF once on each of the four datasets and five ε values, reporting the total variation distances (TVD). Runtime is 2-5 hours depending on hardware.

You may modify `script.py` to run a subset of the experiments.

### 2.2 Generate Synthetic Data

To generate a synthetic dataset with specified settings:

```
python3 main.py
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

Default δ=1e-5. For other values, calculate privacy budget with `cal_privacy_budget()` in `PrivMRF/utils/tools.py` and hardcode the result in `privacy_budget()`.

### 3.1 Attribute Hierarchy

The `adult` dataset experiment bins attribute values to improve noise resistance, as in PrivBayes[1]. This slightly improves PrivMRF performance on `adult` but doesn't always help for other datasets.

To use attribute hierarchy:
1. Define hierarchy (see `data/adult_hierarchy.json`) 
2. Read with `read_hierarchy()` in `PrivMRF/attribute_hierarchy.py`
3. Pass hierarchy to `PrivMRF.run()`  
4. Set `config['enable_attribute_hierarchy'] = True`

## 4. Privacy Budget

Calculate privacy budgets for different ε/δ using `cal_privacy_budget()` in `PrivMRF/utils/tools.py`.

## Code Reference 

https://github.com/ryan112358/private-pgm