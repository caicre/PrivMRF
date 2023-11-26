import numpy as np
import json
from . import utils
import re

class Attribute:
    def __init__(self, attribute):
        self.attribute = attribute  # attribute id, which is equal to the column number for top level attributes
        self.level_to_size = None   # None for undecided, map from level to its size. Only valid for attribute
        self.node_list = []         # specify its children. Note nodes are values in current level as well
        self.value_list = []        # specify values in current level

    def string(self, tab=0):
        res = '\t'*tab+  'id: {}\n'.format(self.attribute) \
            +'\t'*tab +  'level_to_size: {}\n'.format(str(self.level_to_size)) \
            +'\t'*tab +  'value_list: {}\n'.format(str(self.value_list))
        res +='\t'*tab + 'node_list:\n'
        for node in self.node_list:
            res += node.string(tab+1)
        return res

    def generate_node(self, histogram, value_list, attribute_id, bin_num, max_size, depth):
        if depth <= 0:
            return attribute_id
        assert(len(histogram)==len(value_list))
        size = len(value_list)
        histogram_sum = np.sum(histogram)
        current_count = 0
        current_bin_start_value = 0
        current_value = 0
        for i in range(bin_num):
            attribute = Attribute(attribute_id)
            attribute_id += 1
            bin_value_list = []
            current_bin_start_value = current_value

            while True:
                current_count += histogram[current_value]
                bin_value_list.append(value_list[current_value])
                current_value += 1

                if current_count >= histogram_sum * (i+1)/bin_num:
                    break

                if current_value >= size:
                    break

            if i == bin_num - 1 and current_value < size:
                bin_value_list.extend(value_list[current_value:])
            
            self.node_list.append(attribute)
            if len(bin_value_list) > max_size:
                attribute_id = attribute.generate_node(histogram[current_bin_start_value: current_value], \
                    value_list[current_bin_start_value: current_value], attribute_id, bin_num, max_size, depth-1)
            else:
                attribute.value_list = bin_value_list
        
        return attribute_id

# return attribute list after discretization
# each attibute indicates its children and its children may indicate their children
def equal_width_equal_frequency_discretization(data, domain, max_size=1000, bin_num=10, ratio=0):
    data_num = len(data)
    attribute_list = [0] * data.shape[1]
    attribute_id = data.shape[1]
    for i in range(data.shape[1]):
        size = domain.project([i]).shape[0]
        if size < max_size:
            attribute_list[i] = Attribute(i)
            attribute_list[i].value_list = list(range(size))
            continue
        if ratio >= 0:
            histogram, _ = np.histogramdd(data[:, i], bins=[size])
            histogram += data_num * ratio
        else:
            histogram = np.ones(size)

        attribute = Attribute(i)
        attribute_id = attribute.generate_node(histogram, list(range(size)), attribute_id, bin_num, max_size, 2)
        attribute_list[i] = attribute

    return attribute_list

def dump_attribute(attribute):
    info = {'name': attribute.attribute}
    info['node_list'] = [dump_attribute(i) for i in attribute.node_list]
    info['value_list'] = attribute.value_list
    info['level_to_size'] = attribute.level_to_size
    return info

def write_hierarchy(attribute_list, path):
    write_list = []
    for attribute in attribute_list:
        write_list.append(dump_attribute(attribute))
    with open(path, 'w') as out_file:
        json.dump(write_list, out_file)

def load_attribute(info):
    attribute = Attribute(int(info['name']))
    attribute.node_list = [load_attribute(i) for i in info['node_list']]
    attribute.value_list = info['value_list'].copy()
    if info['level_to_size'] is not None:
        attribute.level_to_size = {int(key): value for key, value in info['level_to_size'].items()}
    else:
        attribute.level_to_size = None

    return attribute

def update_level_to_size(node):
    level_to_size = {}

    def visit_node(node, level):
        nonlocal level_to_size
        if level in level_to_size:
            level_to_size[level] += len(node.value_list)
        else:
            level_to_size[level] = len(node.value_list)

        for sub_node in node.node_list:
            visit_node(sub_node, level+1)

    visit_node(node, 0)
    acc = 0
    for i in range(len(level_to_size)):
        level_to_size[i] += acc
        acc = level_to_size[i]

    def visit_node_node(node, level):
        nonlocal level_to_size
        level_to_size[level] += len(node.node_list)

        for sub_node in node.node_list:
            visit_node_node(sub_node, level+1)
    visit_node_node(node, 0)

    node.level_to_size = level_to_size

# convert parentheses string to hierarchy
# you may write your own code to generate hierarchy. This func is 
# specially for PrivBayes data: https://sourceforge.net/projects/privbayes/files/
# If your attributes are small, it might be feasible to write parentheses string like PrivBayes
# then use this func to generate hierarchy
def str_to_hierarchy(data_name, path):
    str_path = './data_/' + data_name + '.domain'
    attribute_list = []
    with open(str_path, 'r') as in_file:
        i = 0
        line_list = []
        for line in in_file:
            attribute_list.append(Attribute(i))
            line_list.append(re.split(r'([ \{\}])', line.strip()))

    node_id = len(line_list)
    def visit_string(string_list, i):
        nonlocal node_id
        # print(string_list, i)
        node = Attribute(node_id)
        node_id += 1
        length = len(string_list)
        while i < length:
            item = string_list[i]
            if item == ' ' or item == '':
                i += 1
                continue
            elif item == '{':
                sub_node, i = visit_string(string_list, i+1)
                node.node_list.append(sub_node)
            elif item == '}':
                i += 1
                break
            else:
                node.value_list.append(item)
            i += 1
        return node, i

    domain = utils.tools.read_json_domain('./data/'+data_name+'.json')

    for i in range(len(line_list)):
        string_list = line_list[i]
        if string_list[0] == 'C':
            attribute = Attribute(i)
            attribute.value_list = [str(temp) for temp in list(range(domain[i]['domain']))]
            attribute.level_to_size = {0: domain[i]['domain']}
            attribute_list[i] = attribute
        else:
            attribute, _ = visit_string(string_list[2:], 0)
            attribute.attribute = i
            attribute_list[i] = attribute
    
    for i in range(len(line_list)):
        update_level_to_size(attribute_list[i])
    
    write_hierarchy(attribute_list, path)

# read attribute hierarchy, no need to be generated by equal_width_equal_frequency_discretization
# a json file storing attribute list, each attibute info is a dictionary
# 'name':       attribute id (column number)
# 'node_list':  children id list (also Attribute type), empty if no children
# 'value_list': speifies values that belongs to this attribute
#               empty if has children, while its values are specified by its children
def read_hierarchy(path):
    with open(path, 'r') as in_file:
        read_list = json.load(in_file)
    attribute_list = []
    for info in read_list:
        attribute = load_attribute(info)
        attribute_list.append(attribute)

    attribute_list.sort(key=lambda x: x.attribute)
    # for attr in attribute_list:
    #     print(attr.string())
    return attribute_list

def dat_to_csv(dat_path, csv_path, delm=' '):
    data_list = []
    with open(dat_path, 'r') as in_file:
        for line in in_file:
            data_list.append(line.strip().split(delm))
    utils.tools.write_csv(data_list, list(range(len(data_list[0]))), csv_path)

def str_to_int_hierarchy(hierarchy, map_list):
    return hierarchy

def max_id_of_node(node):
    max_value = node.attribute
    for sub_node in node.node_list:
        value = max_id_of_node(sub_node)
        if value > max_value:
            max_value = value
    return max_value




# generate hierarchy for big attributes
# attr_list: generate hierarchy for these attribute ids
# hierarchy: input attribute hierarchy list
# dom_size = base_size x factor_size x factor_size x ....
def reset_heirarcy(hierarchy, attr_list, base_size=10, dom_threshold=10):
    new_id = 0
    for node in hierarchy:
        new_id = max(max_id_of_node(node), new_id)
    new_id += 1
    print('new id:', new_id)

    def split_node(node, base_size, new_id, dfs=False):
        sub_node_size = int(len(node.value_list)/base_size)+1
        for i in range(base_size):
            temp_node = Attribute(new_id)
            new_id += 1
            
            temp_node.value_list = node.value_list[i*sub_node_size: (i+1)*sub_node_size]
            node.node_list.append(temp_node)
            if dfs and len(temp_node.value_list) > dom_threshold:
                new_id = split_node(temp_node, base_size, dfs)
        node.value_list = []
        return new_id


    for attr in attr_list:
        attr = hierarchy[attr]
        if len(attr.level_to_size) == 1 and attr.level_to_size[0] > dom_threshold:
            new_id = split_node(attr, base_size, new_id, dfs=True)
            update_level_to_size(attr)

    return hierarchy

# generate value list for original data so that we can set hierarchy manually
# or use reset_hierarchy to rearange the hierachy automatically
def generate_value_list(data, attr_list):
    data_list, headings = utils.tools.read_csv('./data/' + data + '.csv')

    attr_to_value_list = {}
    for attr in attr_list:
        value_set = set()
        for line in data_list:
            if line[attr] not in value_set:
                value_set.add(line[attr])
        value_set = [int(item) for item in list(value_set)]
        value_set.sort()
        value_set = [str(item) for item in value_set]
        attr_to_value_list[attr] = value_set

    with open('./data/value_list.json', 'w') as out_file:
        json.dump(attr_to_value_list, out_file)

def get_hierarchy(read_list):
    attribute_list = []
    for info in read_list:
        attribute = load_attribute(info)
        attribute_list.append(attribute)

    attribute_list.sort(key=lambda x: x.attribute)
    # for attr in attribute_list:
    #     print(attr.string())
    return attribute_list

def get_one_level_hierarchy(domain):
    read_list = []
    for attr, value in domain.dict.items():
        temp_dict = {}
        temp_dict['name'] = 0
        temp_dict['node_list'] = []
        temp_dict['value_list'] = [str(item) for item in range(value['domain'])]
        temp_dict['level_to_size'] = {'0': value['domain']}

        read_list.append(temp_dict)

    return get_hierarchy(read_list)