import os
import json
from .utils import tools
import pickle
import numpy as np
from .domain import Domain
import csv
import random
import pandas as pd
from . import attribute_hierarchy as ah

# convert int data to string data
def data_num_to_str(map_list, np_data):
    print('  convert int data to string data')

    invert_map_list = []
    for str_to_int in map_list:
        invert_map_list.append({value: key for key, value in str_to_int.items()})

    data_list = []
    attr_num = np_data.shape[1]
    for line in np_data:
        data_list.append([invert_map_list[index][line[index]] for index in range(attr_num)])

    return data_list

# for simplicity, convert string data to int data, which can be accepted by models
# and easy to evaluate. String data and int data are equivalent. Use postprocess to
# convert outputs of models to string data if you need.
def preprocess(data):
    path = './preprocess'
    if not os.path.exists(path):
        os.mkdir(path)
    data_list, headings = tools.read_csv('./data/' + data + '.csv')
    # json_domain = tools.read_json_domain('./data/' + data + '.json')
    attr_list = ah.read_hierarchy('./data/'+data+'_hierarchy.json')

    attr_num = len(headings)
    json_domain = {attr: {'type': 'discrete'} for attr in range(attr_num)}

    max_level = []
    for attr in range(attr_num):
        max_level.append(max(attr_list[attr].level_to_size.keys()))

    # value: str to int
    value_id_to_vn = [0]*attr_num
    value_to_value_id = [0]*attr_num

    for attr in range(attr_num):
        value_set = set()
        for line in data_list:
            if line[attr] not in value_set:
                value_set.add(line[attr])

        value_list = list(value_set)
        value_list.sort()

        temp_value_to_value_id = {}
        temp_value_id_to_value = {}
        for value_id in range(len(value_set)):
            temp_value_to_value_id[value_list[value_id]] = value_id
            temp_value_id_to_value[value_id] = value_list[value_id]
        value_to_value_id[attr] = temp_value_to_value_id
        value_id_to_vn[attr] = temp_value_id_to_value

        json_domain[attr]['domain'] = len(value_set)
    domain = Domain(json_domain, list(range(attr_num)))

    # hierarchy map: str to int
    def visit_value_list(attr, value_to_value_id):
        # print(value_to_value_id, attr.value_list)
        attr.value_list = [value_to_value_id[value] for value in attr.value_list if value in value_to_value_id]
        for node in attr.node_list:
            visit_value_list(node, value_to_value_id)

    for attr in range(attr_num):
        visit_value_list(attr_list[attr], value_to_value_id[attr])
        # print(attr_list[attr].string())

    print('attrs: ', domain.attr_list)
    print('shape: ', domain.shape)

    # data: str to int
    new_data_list = []
    for line in data_list:
        new_line = [value_to_value_id[attr][line[attr]] for attr in range(attr_num)]
        new_data_list.append(new_line)

    json.dump(value_id_to_vn, open('./preprocess/' + data + '_int_to_str.json', 'w'))
    json.dump(value_to_value_id, open('./preprocess/' + data + '_str_to_int.json', 'w'))

    tools.write_csv(new_data_list, headings, './preprocess/' + data + '.csv')
    json.dump(json_domain, open('./preprocess/' + data + '.json', 'w'))
    ah.write_hierarchy(attr_list, './preprocess/' + data + '_hierarchy.json')

    data = np.array(data_list)

    return data, domain, attr_list

def read_preprocessed_data(data, task='TVD'):
    if task == 'TVD':
        data_list, headings = tools.read_csv('./preprocess/' + data + '.csv')
        json_domain = tools.read_json_domain('./preprocess/' + data + '.json')
        attr_list = ah.read_hierarchy('./preprocess/' + data + '_hierarchy.json')
    else:
        data_list, headings = tools.read_csv('./exp_data/' + data + '_train.csv')
        json_domain = tools.read_json_domain('./preprocess/' + data + '.json')
        attr_list = ah.read_hierarchy('./preprocess/' + data + '_hierarchy.json')

    data = np.array(data_list, dtype=int)
    domain = Domain(json_domain, list(range(data.shape[1])))

    return data, domain, attr_list

# In order to compare with PrivBayes, we use the same way to preprocess
# continuous attribute. We convert them to discrete attributes, each of which
# has a domain size of 16.
def convert_continuous_attribute_to_discrete(
    data, out_path, out_json_path, out_hierarchy_path, bin_num=16):
    if data == 'br2000':
        continuous_attrs = [4, 10, 12]
    elif data == 'adult':
        continuous_attrs = [0, 2, 10, 11, 12]

    data_list, headings = tools.read_csv('./data/' + data + '.csv')
    for attr in continuous_attrs:
        value_set = set()
        for line in data_list:
            if line[attr] not in value_set:
                value_set.add(line[attr])

        value_list = [int(item) for item in list(value_set)]
        value_list.sort()
        min_value = value_list[0]
        bin_size = (value_list[-1] - min_value)/bin_num

        value_map = {}
        value_id = 0
        for value in value_list:
            while value > min_value + (value_id+1) * bin_size + 1:
                value_id += 1
            value_map[value] = value_id
        
        for line in data_list:
            line[attr] = value_map[int(line[attr])]

    tools.write_csv(data_list, headings, out_path)

    with open('./data/' + data + '.json', 'r') as in_file:
        domain = json.load(in_file)
    for attr in continuous_attrs:
        domain[str(attr)]['domain'] = bin_num
    with open(out_json_path, 'w') as out_file:
        json.dump(domain, out_file)

    attr_list = ah.read_hierarchy('./data/'+data+'_hierarchy.json')
    for attr in continuous_attrs:
        attr_list[attr].value_list = list(range(bin_num))
        attr_list[attr].level_to_size[0] = bin_num
    ah.write_hierarchy(attr_list, out_hierarchy_path)

# int to str
def postprocess(in_path, out_path):
    print('postprocess data')

    value_id_to_vn = json.load(open('./preprocess/' + data + '_int_to_str.json', 'w'))
    data_list, headings = tools.read_csv(in_path)

    new_data_list = []
    for line in data_list:
        new_line = [value_id_to_vn[attr][line[attr]] for attr in range(len(headings))]
        new_data_list.append(line)

    tools.write_csv(new_data_list, headings, out_path)

if __name__ == '__main__':
    pass
    # def func(data_name):
    #     data_list, headings = tools.read_csv('./data/'+data_name+'.csv')
    #     attr_num = len(headings)

    #     attr_list = []
    #     for i in range(attr_num):
    #         attr = ah.Attribute(i)
    #         attr.level_to_size = {0: 2}
    #         attr.value_list = ["0", "1"]
    #         attr_list.append(attr)

    #     ah.write_hierarchy(attr_list, './data/'+data_name+'_hierarchy.json')


    # preprocess('acs')
    # preprocess('nltcs')
    # preprocess('br2000')
    # preprocess('adult')