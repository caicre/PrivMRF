# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

import contextlib
import os
import time

from .attribute_graph import AttributeGraph
from .attribute_hierarchy import get_one_level_hierarchy
from .backend import gpu_available
from .markov_random_field import MarkovRandomField
from .utils import tools

DEFAULT_CONFIG = {
    'data':                         'no_data_name', # only used to name debug outputs
    'delta':                        1e-5,
    'theta':                        6,
    'print':                        True,
    'gpu':                          None,   # None: use GPU when cupy and a GPU are available
    'log_dir':                      None,   # directory for debug outputs (graph, measures, log), None to skip them

    # privacy budget allocation (Section 3.6 of the paper),
    # the rest 1 - beta1 - beta3 - beta_n is for marginal distributions
    'beta1':                        0.10,   # R-scores for constructing the attribute graph
    'beta3':                        0.10,   # L1 norm queries for selecting marginals
    'beta_n':                       0.01,   # noisy number of records
    't':                            0.8,    # select t * attr_num marginals when refining the marginal set
    'norm_query_number':            400,    # k, the number of marginals sampled in each selection

    'init_measure':                 0,      # 0 inner Bayesian Network
                                            # 1 all 2 way measure
                                            # 2 clique measure
                                            # 3 empty measure
    'supplement_2way':              False,
    'enable_attribute_hierarchy':   False,
    'attr_level':                   None,   # {attr: level}, overrides the initial hierarchy level of attributes
    'last_estimation':              False,
    'max_level_gap':                1,

    'estimation_iter_num':          3000,
    'print_interval':               50,

    'max_clique_size':              1e7,
    'max_parameter_size':           5e7,
    'size_penalty':                 1e-8,

    'max_measure_attr_num':         6,

    'convergence_ratio':            1.2,
    'final_convergence_ratio':      0.7,
}

def run(data, domain, attr_hierarchy=None, exp_name='exp', epsilon=0.8, p_config=None):
    config = DEFAULT_CONFIG.copy()
    if p_config is not None:
        config.update(p_config)

    config['theta1'] = config['theta']
    config['theta2'] = config['theta']
    config['epsilon'] = epsilon
    config['exp_name'] = 'PrivMRF_'+ config['data'] + '_' + exp_name
    config['privacy_budget'] = tools.privacy_budget(epsilon, config['delta'])
    if config['gpu'] is None:
        config['gpu'] = gpu_available()
    if config['log_dir'] is not None:
        os.makedirs(config['log_dir'], exist_ok=True)
    if attr_hierarchy is None:
        attr_hierarchy = get_one_level_hierarchy(domain)

    if config['print']:
        return train(data, domain, attr_hierarchy, config)

    if config['log_dir'] is None:
        log_path = os.devnull
    else:
        log_path = os.path.join(config['log_dir'], config['exp_name'] + '_log.txt')
    with open(log_path, 'w') as log_file, contextlib.redirect_stdout(log_file):
        return train(data, domain, attr_hierarchy, config)

def train(data, domain, attr_hierarchy, config):
    print('PrivMRF')
    print('theta:', config['theta'])
    start_time = time.time()

    init_model = AttributeGraph(data, domain, attr_hierarchy, config)
    graph, measure_list, attr_hierarchy, attr_to_level = init_model.construct_model()

    model = MarkovRandomField(data, domain, graph, measure_list, \
        attr_hierarchy, attr_to_level, init_model.data_num, config, gpu=config['gpu'])
    model.entropy_descent()

    if config['last_estimation']:
        model.config['convergence_ratio'] = 1.0
        model.iter_num = 5000
        model.mirror_descent()

    print('time cost: {:.4f}s'.format(time.time() - start_time))
    return model
