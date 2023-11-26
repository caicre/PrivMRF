import PrivMRF
import PrivMRF.utils.tools as tools
from PrivMRF.domain import Domain
import numpy as np

if __name__ == '__main__':
    # should provide int data
    data, _ = tools.read_csv('./preprocess/nltcs.csv')
    data = np.array(data, dtype=int)

    # domain of each attribute should be [0, 1, ..., max_value-1]
    # attribute name should be 0, ..., column_num-1.
    json_domain = tools.read_json_domain('./preprocess/nltcs.json')
    domain = Domain(json_domain, list(range(data.shape[1])))

    # you may set hyperparameters or specify other settings here
    config = {
    }

    # train a PrivMRF, delta=1e-5
    # for other dp parameter delta, calculate the privacy budget 
    # with cal_privacy_budget() of ./PrivMRF/utils/tools.py 
    # and hard code the budget in privacy_budget() of ./PrivMRF/utils/tools.py 
    model = PrivMRF.run(data, domain, attr_hierarchy=None, \
        exp_name='exp', epsilon=0.8, p_config=config)

    # generate synthetic data
    syn_data = model.synthetic_data('./out.csv')