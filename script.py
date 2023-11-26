import os

# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from exp.evaluate import run_experiment, split
import json
import csv
import pandas as pd
import numpy as np

import sys
from PrivMRF.preprocess import preprocess

# python3 script.py adult 1

if __name__ == '__main__':
    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)

    preprocess('acs')
    preprocess('nltcs')
    preprocess('br2000')
    preprocess('adult')

    # adult, br2000, nltcs, acs
    # data_list = ['br2000']
    data_list = ['nltcs', 'acs', 'adult', 'br2000']

    # PrivMRF
    method_list = ['PrivMRF']

    # arbitrary string for naming output data
    exp_name = 'test'

    # 0.1, 0.2, 0.4, 0.8, 1.6, 3.2
    # epsilon_list = [1.6]
    epsilon_list = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

    # number of experiments
    repeat = 1

    run_experiment(data_list, method_list, exp_name, task='TVD', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25)

    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25)
    # run_experiment(data_list, method_list, exp_name, task='SVM', epsilon_list=epsilon_list, repeat=repeat, classifier_num=25, generate=False)

