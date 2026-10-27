import PrivMRF
from PrivMRF.preprocess import read_preprocessed_data

# settings of the experiments in the paper
# Data synthesis via differentially private Markov random fields
def dataset_config(data_name, epsilon):
    config = {'data': data_name, 'log_dir': './temp'}

    if data_name in ('acs', 'nltcs'):
        # acs is too large for the GPU memory, and nltcs is small enough for CPU
        config['gpu'] = False
    if data_name == 'acs':
        config['size_penalty'] = 1e-7
        config['max_clique_size'] = 2e5
        config['max_parameter_size'] = 1e6
    if data_name == 'adult':
        config['enable_attribute_hierarchy'] = True
        if epsilon < 0.41:
            config['max_parameter_size'] = 3e7
        if epsilon <= 0.21:
            config['attr_level'] = {3: 0, 13: 1}
        elif epsilon <= 0.41:
            config['attr_level'] = {3: 1, 13: 1}

    return config

def run_syn(data_name, exp_name, epsilon, task='TVD'):
    data, domain, attr_hierarchy = read_preprocessed_data(data_name, task)
    model = PrivMRF.run(data, domain, attr_hierarchy, exp_name, epsilon, dataset_config(data_name, epsilon))
    return model.synthetic_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
