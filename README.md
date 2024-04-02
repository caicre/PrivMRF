# Data Synthesis via Differentially Private Markov Random Fields

Kuntai Cai, Xiaoyu Lei, Jianxin Wei, Xiaokui Xiao

caikt@comp.nus.edu.sg

This project provides the implementation of "Data Synthesis via Differentially Private Markov Random Fields". You can generate a synthetic dataset with PrivMF and reproduce the experimental results using a single command.


## 1. Get Started

These codes require Python3.8, cuda11.7 and need to run on a GPU that supports `cupy`. The dependencies can be installed via

    $ pip3 install -r requirements.txt

After that, you can simply reproduce the experimental results reported in the paper via

    $ python3 script.py

By default, it runs PrivMRF once on each of the four datasets and five values of $\epsilon$ and reports the total variation distances (TVD). It takes two to five hours to complete, depending on the performance of the computer. You may modify the codes in script.py to run PrivMRF only in some cases.

You can also run PrivMRF only once with a specified setting via

    $ python3 main.py

This will generate a synthetic dataset but it does not report any result.

## 2. Reproduce the SVM Results

For simplicity, we comment out the codes that run SVM experiments in script.py. You can uncomment them to reproduce SVM experiments.

    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25)
    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25, generate=False)

The first line generates synthetic data. The second line trains SVM classifiers on the synthetic data and tests their mis-classification rates, which is a parallel program. Each combination of a dataset, method, and epsilon needs one process. You may want to avoid calling too many processes by setting `data_list, method_list, epsilon_list` such that their sizes are small. Training and testing SVMs for one dataset and one $\epsilon$ take 1 to 6 hours.

## 3. Run PrivMRF on Other Datasets

`./main.py` provides a simple example for runing PrivMRF with the default config. You just need to read your data and call PrivMRF to train the model and generate a synthetic version of the data. Note that you must preprocess your data and the domain such that they only contain discete int values. Note that by default we set $\delta=1e-5$. For a different $\delta$ value, You may calculate the privacy budget with cal_privacy_budget() of `./PrivMRF/utils/tools.py` and hard code the budget in `privacy_budget()` of `./PrivMRF/utils/tools.py`.


### 3.1 Attribute Hierarchy

In the `adult` dataset, for each attribute, we merge values into bins. Since the number of records in a bin is larger than the number of records of each value of the bin, merging values provides more resistance to noise. For the `adult` dataset, PrivMRF with this technique provides slightly better performance. PrivBayes[1] provides a detailed description of attribute hierarchy. For Makov random field, we apply attribute hierarchy to marginal distributions. That is, we only count the number of records in each bin instead of counting the number of records of each possible combination of values.

Attribute hierarchy does not always work for PrivMRF. You can simply ignore this technique. However, if you want to try this technique
*   See `./data/adult_hierarchy.json` for the example of attribute hierarchy.
*   Use `read_hierarchy()` in `./PrivMRF/attribute_hierarchy.py` to read a hierarchy
*   Send the hierarchy to `PrivMRF.run()` as a parameter.
*   Set config['enable_attribute_hierarchy'] = True and send the config to `PrivMRF.run()`.

## 4. Calculating Privacy Budget

The privacy budgets are calculated via `cal_privacy_budget()` in `./PrivMRF/utils/tools.py`. You can call this function to calculate the privacy budgets of different $\epsilon$ and $\delta$.

## Code Reference

https://github.com/ryan112358/private-pgm