import os

# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

# thread number for numpy (when it runs on CPU)
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

from .preprocess import read_preprocessed_data, postprocess
from .attribute_graph import AttributeGraph
from .attribute_hierarchy import get_one_level_hierarchy
from .markov_random_field import MarkovRandomField
from networkx.readwrite import json_graph
import json
import numpy as np
import pickle
import time
import sys
from .preprocess import preprocess

def run(data, domain, attr_hierarchy, exp_name, epsilon, task='TVD', p_config=None):
    default_config = {

        'beta5':        0.00,   # construct inner Bayesian network
        'data':         'no_data_name',
        
        'theta':        6,
        'print':        True,

        'score':        'pairwsie_TVD', # pairwsie_TVD is emperically better
        # 'score':        'pairwsie_MI',
        # 'score':        'pairwise_entropy',
 
        'score_R':                      False,
        'init_measure':                 0, # 0 inner Bayesian Network 
                                        # 1 all n way measure
                                        # 2 clique measure
                                        # 3 empty measure
        'supplement_2way':              False,
        'attr_measure':                 False,
        'enable_attribute_hierarchy':   False,
        # 'enable_attribute_hierarchy':   True,
        'last_estimation':              False,
        'init_model':                   True,
        'max_level_gap':                1,

        'estimation_iter_num':          3000,
        'print_interval':               50,

        'max_clique_size':              1e7,
        'max_parameter_size':           5e7,
        'size_penalty':                 1e-8,

        'estimation_method':            'mirror_descent',

        'max_measure_attr_num':         6,
        'max_measure_attr_num_privBayes':5,

        'convergence_ratio':            1.2,
        'final_convergence_ratio':      0.7,

        'use_exp_mech':                 -1,      # do not use exponential mechanism to select measures
        # 'use_exp_mech':                 0.05,
        'structure_entropy':            False,   # marginal_noise will be set 0 to calculate the entropy of structures

        'noise_type':                   'normal' # only support normal
    }

    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)

    if default_config['use_exp_mech'] > 0:
        default_config['beta1'] = 0.12 # dependency graph, Markov network
        default_config['beta2'] = 0.55 # marginal distributions of initial marginals
        default_config['beta4'] = 0.33 # marginal distributions of newly selected marginals
    else:
        if default_config['init_measure'] == 3:
            default_config['beta1'] = 0.10 
            default_config['beta2'] = 0.0
            default_config['beta3'] = 0.10
            default_config['beta4'] = 0.80
        else:
            default_config['beta1'] = 0.10 # dependency graph, Markov network
            default_config['beta2'] = 0.50 # marginal distributions of initial marginals
            default_config['beta3'] = 0.10 # query L_1 norms
            default_config['beta4'] = 0.30 # marginal distributions of newly selected marginals

            default_config['t'] = 0.8
            # beta2, beta4 is no longer uesful, we use t to allocate budget for marginal distribution
            # we ensure that beta2 + beta4 = 1 - (beta1 + beta3)

    config = default_config.copy()
    if p_config is not None:
        for item in p_config:
            config[item] = p_config[item]

    if not config['print']:
        temp_stream = sys.stdout
        sys.stdout = open('./temp/log.txt', 'w')

    # There might be no enough resource to run PrivMRF on GPU
    # acs should be runned on cpu, nltcs is too small and doesn't have to be runned on GPU
    gpu = False
    if config['data'] == 'adult' or config['data'] == 'br2000':
        gpu = True
    if config['data'] == 'acs' or config['data'] == 'nltcs':
        default_config['max_measure_attr_num'] = 10
        default_config['max_measure_attr_num_privBayes'] = 9
    if config['data'] == 'adult':
        default_config['enable_attribute_hierarchy'] = True

    config['theta1'] = config['theta']
    config['theta2'] = config['theta']
    config['epsilon'] = epsilon
    config['exp_name'] = 'PrivMRF_'+ config['data'] + '_' + exp_name
    if attr_hierarchy is None:
        attr_hierarchy = get_one_level_hierarchy(domain)
    print('PrivMRF')


    start_time = time.time()
    
    print('theta:', config['theta'])
    if config['init_model']:
        init_model = AttributeGraph(data, domain, attr_hierarchy, config, config['data'])
        graph, measure_list, attr_hierarchy, attr_to_level, entropy = init_model.construct_model()
        # AttributeGraph.save_model(init_model, './temp/' + config['data'] + '_model.mdl')

    # return entropy

    # init_model = AttributeGraph.load_model('./temp/' + config['data'] + '_model.mdl')
    graph = init_model.graph
    measure_list = init_model.measure_list
    attr_hierarchy = init_model.attr_list
    attr_to_level = init_model.attr_to_level
    data_num = init_model.data_num

    model = MarkovRandomField(data, domain, graph, measure_list, \
        attr_hierarchy, attr_to_level, data_num, config, gpu=gpu)
    model.entropy_descent()
    # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_model.mrf')

    # model = MarkovRandomField.load_model('./temp/' + config['data'] + '_model.mrf')
    if config['last_estimation']:
        model.config['convergence_ratio'] = 1.0
        model.config['estimation_iter_num'] = 5000
        model.mirror_descent()
        # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_le_model.mrf')

    
    time_cost = time.time() - start_time
    print('time cost: {:.4f}s'.format(time_cost))

    if not config['print']:
        sys.stdout.close()
        sys.stdout = temp_stream

    os.chdir(cwd)

    return model

# used for experiments of the paper Data synthesis via differentially private Markov random field
def run_syn(data_name, exp_name, epsilon, task='TVD'):
    p_config = {}
    p_config['data'] = data_name
    
    data, domain, attr_hierarchy = read_preprocessed_data(data_name, task)
    model = run(data, domain, attr_hierarchy, exp_name, epsilon, task, p_config)
    data_list = model.synthetic_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
    return data_list

