from functools import reduce
import numpy as np

class Domain:
    # attr_list specifies the order of axis
    def __init__(self, domain_dict, attr_list):
        self.dict = domain_dict
        self.attr_list = attr_list
        self.shape = [domain_dict[i]['domain'] for i in attr_list]


    def project(self, attr_set):
        new_dict = {key: self.dict[key] for key in attr_set}
        new_attr_list = [attr for attr in self.attr_list if attr in attr_set]
        return Domain(new_dict, new_attr_list)
    
    def moveaxis(self, attr_list):
        attr_set = set(self.attr_list)
        new_attr_list = [attr for attr in attr_list if attr in attr_set]
        return Domain(self.dict, new_attr_list)

    def attr_domain(self, attr):
        if attr in self.dict:
            return self.dict[attr]['domain']
        else:
            return None

    def size(self):
        return reduce(lambda x,y: x*y, self.shape, 1)
    
    # edge for np.histogramdd
    def edge(self):
        return [list(range(i+1)) for i in self.shape]

    def index_list(self, domain):
        if not isinstance(domain, Domain):
            attr_list = domain
        else:
            attr_list = domain.attr_list
        index_list = []
        for attr in attr_list:
            index_list.append(self.attr_list.index(attr))
        return index_list

    def invert(self, domain):
        new_dict = {}
        new_attr_list = []
        for i in self.attr_list:
            if i not in domain.dict:
                new_attr_list.append(i)
                new_dict[i] = self.dict[i]
        return Domain(new_dict, new_attr_list)

    def equal(self, domain):
        if len(self.attr_list) != len(domain.attr_list):
            return False
        for i in range(len(self.attr_list)):
            if self.attr_list[i] != domain.attr_list[i]:
                return False
        return True

    def __sub__(self, parameter):
        domain = [attr for attr in self.dict if attr not in parameter.dict]
        return self.project(domain)

    def __add__(self, parameter):
        domain_dict = self.dict.copy()
        for attr in parameter.dict:
            domain_dict[attr] = parameter.dict[attr]
        attr_list = self.attr_list.copy()
        for attr in parameter.attr_list:
            if attr in set(parameter.attr_list) - set(self.attr_list):
                attr_list.append(attr)
        return Domain(domain_dict, attr_list)

    def __len__(self):
        return len(self.dict)

class Smoother:
    def __init__(self, histogram, domain, value_range, value_threshold):
        self.domain = domain
        histogram = histogram.flatten()
        values, indices, counts = np.unique(histogram, return_index=True, return_counts=True)
        value_counts = list(zip(values, indices, counts))
        value_counts.sort(key=lambda x: x[0])

        min_range = value_counts[0][2]
        current_range = min_range + value_range
        
        self.index_map = {}
        self.new_index_id = 0
        for i in range(len(value_counts)):
            if value_counts[i][0] > value_threshold:
                break
            if value_counts[i][0] < current_range:
                self.index_map[value_counts[i][1]] = self.new_index_id
            else:
                while value_counts[i][0] >= current_range:
                    current_range += value_range
                self.new_index_id += 1
                self.index_map[value_counts[i][1]] = self.new_index_id
        print('compress marginal domain', domain.attr_list, self.new_index_id, domain.size())

    def smoothed_noisy_histogram(self, histogram, domain, noise):
        if not self.domain.equal(domain):
            print('error: wrong smoother')
            exit(-1)
        histogram = histogram.flatten()
        values, indices, counts = np.unique(histogram, return_index=True, return_counts=True)
        
        new_index_to_value = {index: 0 for index in range(self.new_index_id+1)}
        new_index_to_index = {index: [] for index in range(self.new_index_id+1)}
        for index in self.index_map:
            new_index = self.index_map[index]
            new_index_to_value[new_index] += histogram[index]
            new_index_to_index[new_index].append(index)
        
        histogram += np.random.normal(scale=noise, size=domain.size())

        for new_index in new_index_to_index:
            value = new_index_to_value[new_index] + np.random.normal(scale=noise)
            indices_num = len(new_index_to_index[new_index])
            for index in new_index_to_index[new_index]:
                histogram[index] = int(value/indices_num)

        histogram = np.reshape(histogram, domain.shape)

        return histogram