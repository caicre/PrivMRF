import json

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

# the domain size of the subattribute when the attribute is used at the level,
# i.e., the number of values of the largest node at the level
def subattr_size(attribute, level):
    def nodes_at(node, depth):
        if depth == level:
            return node.node_list
        return [item for child in node.node_list for item in nodes_at(child, depth+1)]

    def value_num(node):
        return len(node.value_list) + sum(value_num(child) for child in node.node_list)

    return max((value_num(node) for node in nodes_at(attribute, 0)), default=0)

# the number of cells an attribute contributes to a marginal when it is used at the level,
# which is (level size) * (subattribute size) if the level is not the finest one
def expanded_size(attribute, level):
    size = attribute.level_to_size[level]
    if level < max(attribute.level_to_size.keys()):
        size *= subattr_size(attribute, level)
    return size

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

# read attribute hierarchy
# a json file storing attribute list, each attibute info is a dictionary
# 'name':       attribute id (column number)
# 'node_list':  children id list (also Attribute type), empty if no children
# 'value_list': speifies values that belongs to this attribute
#               empty if has children, while its values are specified by its children
def read_hierarchy(path):
    with open(path, 'r') as in_file:
        read_list = json.load(in_file)
    return get_hierarchy(read_list)

def get_hierarchy(read_list):
    attribute_list = []
    for info in read_list:
        attribute = load_attribute(info)
        attribute_list.append(attribute)

    attribute_list.sort(key=lambda x: x.attribute)
    return attribute_list

def get_one_level_hierarchy(domain):
    read_list = []
    for attr, value in domain.dict.items():
        temp_dict = {}
        temp_dict['name'] = attr
        temp_dict['node_list'] = []
        temp_dict['value_list'] = [str(item) for item in range(value['domain'])]
        temp_dict['level_to_size'] = {'0': value['domain']}

        read_list.append(temp_dict)
    return get_hierarchy(read_list)
