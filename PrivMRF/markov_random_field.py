from .factor import Factor, Potential
import numpy as np
import cupy as cp
import networkx as nx
from .utils import tools
import math
import itertools
import time
import pickle
import random
import pandas as pd
from .domain import Domain
from .attribute_hierarchy import Attribute
import os
import json


class MarkovRandomField:
    def __init__(self, data, domain, graph, measure_list, attr_list, \
        attr_to_level, noisy_data_num, config, gpu=True):

        self.data = data
        self.domain = domain
        self.attr_list = attr_list
        self.attr_to_level = attr_to_level
        self.config = config
        self.data_num = len(self.data)
        self.noisy_data_num = int(noisy_data_num)
        if self.config['structure_entropy']:
            self.noisy_data_num = self.data_num
        self.max_measure_attr_num = config['max_measure_attr_num']
        self.attr_num = len(domain)
        self.gpu = gpu

        # print(mempool.get_limit())

        # triangulate graph        
        if nx.is_chordal(graph):
            self.graph = graph
        else:
            self.graph = tools.triangulate(graph)

        if self.config['enable_attribute_hierarchy']:
            self.improve_attribute_level_for_data_domain()
            self.map_data_with_hierarchy()

            # add subattr to graph
            for attr in range(self.attr_num):
                if self.attr_to_level[attr] == self.max_level[attr]:
                    continue
                subattr = self.attr_to_subattr[attr]
                for neighbor in self.graph[attr]:
                    self.graph.add_edge(subattr, neighbor) 
                self.graph.add_edge(attr, subattr)

        # construct junction tree
        self.maximal_cliques = [tuple(sorted(clique)) for clique in nx.find_cliques(self.graph)]
        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(self.maximal_cliques)
        for clique1, clique2 in itertools.combinations(self.maximal_cliques, 2):
            clique_graph.add_edge(clique1, clique2, weight=-len(set(clique1) & set(clique2)))
        self.junction_tree = nx.minimum_spanning_tree(clique_graph)
        if self.gpu:
            xp = cp
        else:
            xp = np
        self.potential = Potential({clique: Factor.zeros(self.domain.project(clique), xp) for clique in self.maximal_cliques})
        size = sum(self.domain.project(clique).size() for clique in self.maximal_cliques)
        print('model size: {:.4e}'.format(size))
        print('noisy data num: {} data num: {}'.format(self.noisy_data_num, self.data_num))

        print('clique length:', [len(clique) for clique in self.maximal_cliques])
        
        print('convergence_ratio', self.config['convergence_ratio'])
        print('final_convergence_ratio', self.config['final_convergence_ratio'])

        # construct measures
        self.measure_set = measure_list
        if len(self.measure_set) > self.config['init_measure_num']:
            random.shuffle(self.measure_set)
            self.measure_set = set(self.measure_set[:self.config['init_measure_num']])
        else:
            self.measure_set = set(self.measure_set)
        self.config['norm_query_number'] = 400
        print('k:', self.config['norm_query_number'])
        if self.config['use_exp_mech'] > 0:
            self.config['norm_query_number'] = int(1e4)

        temp_list = []
        for measure in self.measure_set:
            temp_list.append(tuple(sorted(list(measure))))
        self.measure_set = set(temp_list)

        if self.config['enable_attribute_hierarchy']:
            for attr in range(self.attr_num):
                if self.attr_to_level[attr] != self.max_level[attr]:
                    print('attr heirarchy measure', (attr, ))
                    self.measure_set.add((attr, ))
        # measure_set contains: single attr, single attr with subattr, multiple attrs

        print('initial measure number: {}'.format(len(self.measure_set)))
        print('initial measure set:')
        if len(self.measure_set) != 0:
            max_len = max([len(measure) for measure in self.measure_set])
            for i in range(1, max_len+1):
                print_list = [item for item in self.measure_set if len(item) == i]
                if len(print_list) > 0:
                    print('    ', print_list)
        if len(self.measure_set) < self.config['init_measure_num']:
            delta_beta = config['beta2'] * (self.config['init_measure_num'] - len(self.measure_set))/self.config['init_measure_num']
            print(len(self.measure_set), self.config['init_measure_num'], delta_beta)
            config['beta2'] -= delta_beta
            config['beta4'] += delta_beta
            print('adjust privacy budget: beta2: {:.2f} beta4: {:.2f}'.format(config['beta2'], config['beta4']))

        self.print_measure_count(self.measure_set)

        # generate noise scale
        self.generte_noise_scale()

        # add subattr for multiple attrs measure
        if self.config['enable_attribute_hierarchy']:
            temp_list = []
            for measure in self.measure_set:
                temp_list.append(\
                    self.get_measure_with_hierarchy(measure))
            self.measure_set = set(temp_list)
            print('  initial measure set with hierarchy:')
            for measure_item in self.measure_set:
                if len(measure_item[1]) != 0:
                    print('    ', measure_item)
        
        if self.config['enable_attribute_hierarchy']:
            measure_dict, measure_set = self.query_marginal_with_hierarchy(self.measure_set)
            self.measure_set = measure_set
            print('  measure set', self.measure_set)
        else:
            measure_dict = tools.dp_marginal_list(self.data, domain, self.measure_set, self.marginal_noise, noise_type=self.config['noise_type'])
            if gpu:
                for measure in measure_dict:
                    temp = measure_dict[measure]
                    measure_dict[measure] = Factor(temp.domain, temp.values, cp)

        self.measure = Potential(measure_dict)
        self.marginal = None

        self.iter_num = config['estimation_iter_num']
        
        # assign measures to maximal cliques
        self.clique_to_measure = {clique: [] for clique in self.maximal_cliques}
        self.measure_to_clique = {}
        for measure in self.measure_set:
            for clique in self.maximal_cliques:
                if set(measure) <= set(clique):
                    self.clique_to_measure[clique].append(measure)
                    self.measure_to_clique[measure] = clique
                    break
        
        self.generte_noise_scale_de()
        # print('measure set', self.measure_set)

        # generate message orders
        message_list = [(a, b) for a, b in self.junction_tree.edges()] + [(b, a) for a, b in self.junction_tree.edges()] 
        message_edge = []
        for message1 in message_list:
            for message2 in message_list:
                if message1[0] == message2[1] and message2[0] != message1[1]:
                    message_edge.append((message2, message1))
        G = nx.DiGraph()
        G.add_nodes_from(message_list)
        G.add_edges_from(message_edge)
        self.message_order = list(nx.topological_sort(G))

        self.average_it = 0


    def set_zero_for_hierarchy(self, measure_value, measure, attrs):
        for attr in attrs:
            subattr = self.attr_to_subattr[attr]
            index1 = measure.index(attr)
            index2 = measure.index(subattr)

            for value in range(self.domain.attr_domain(attr)):
                item = self.value_id_to_vn[attr][value]
                if isinstance(item, Attribute):
                    subattr_range = list(range(len(item.value_list), self.domain.attr_domain(subattr)))
                else:
                    subattr_range = list(range(1, self.domain.attr_domain(subattr)))

                indices = [slice(None)]*measure_value.ndim
                indices[index1] = value
                indices[index2] = subattr_range

                measure_value[tuple(indices)] = 0

    def query_marginal_with_hierarchy(self, measure_set):
        marginal_dict = {}
        temp_list = []
        for measure_item in measure_set:
            attrs = list(measure_item[0])
            subattrs = list(self.attr_to_subattr[attr] for attr in measure_item[1])
            measure = tuple(attrs + subattrs)
            measure_value = tools.dp_marginal_list(self.data, self.domain, \
                [measure], self.marginal_noise, return_factor=False, noise_type=self.config['noise_type'])[measure]

            # temp_domain = self.domain.project(measure)
            # histogram, _ = np.histogramdd(self.data[:, measure], bins=temp_domain.edge())
            # tvd1 = np.sum(np.abs(measure_value - histogram))/self.noisy_data_num/2

            self.set_zero_for_hierarchy(measure_value, measure, measure_item[1])

            # tvd2 = np.sum(np.abs(measure_value - histogram))/self.noisy_data_num/2
            # print('structural zeros', measure, tvd1, tvd2)

            if self.gpu:
                xp = cp
            else:
                xp = np
            
            temp_list.append(measure)
            fact = Factor(self.domain.project(measure), measure_value, xp)
            marginal_dict[measure] = fact

        return marginal_dict, set(temp_list)
    
    # add subattr for measure when possible
    # return (measure, attrs_with_subattr)
    def get_measure_with_hierarchy(self, measure):
        if len(measure) == 2:
            dom_limit = self.max_measure_dom_2way
        elif len(measure) == 1:
            dom_limit = 1e6
        else:
            dom_limit = self.max_measure_dom_high_way
        temp_measure = list(measure)
        random.shuffle(temp_measure)
        res = []
        measure_size = self.domain.project(measure).size()
        for attr in temp_measure:
            if self.attr_to_level[attr] == self.max_level[attr]:
                continue
            temp_size = measure_size/self.attr_list[attr].level_to_size[self.attr_to_level[attr]]\
                *self.attr_list[attr].level_to_size[self.max_level[attr]]

            if temp_size < dom_limit:
                res.append(attr)
                measure_size = temp_size
        # if len(res) > 0:
        #     print('  upgraded measure: ', measure, res)
        return (measure, tuple(sorted(res)))

    # change data and domain to fit attribute heirarchy
    # generate map from string data to int data
    def improve_attribute_level_for_data_domain(self):
        for attr in range(self.attr_num):
            self.domain.dict[attr]['domain'] = \
                self.attr_list[attr].level_to_size[self.attr_to_level[attr]]

        self.max_level = []
        for attr in range(self.attr_num):
            self.max_level.append(max(self.attr_list[attr].level_to_size.keys()))

        # for attr in range(self.attr_num):
        #     print(self.attr_list[attr].string())

        def value_to_attr_value(attribute, cur_level, level, value_id, value_id_to_vn, vn_to_value_id):
            if cur_level <= level:
                for value in attribute.value_list:
                    value_id_to_vn[value_id] = value
                    vn_to_value_id[value] = value_id
                    value_id += 1
            if cur_level < level:
                for node in attribute.node_list:
                    value_id = value_to_attr_value(node, cur_level+1, level, value_id, value_id_to_vn, vn_to_value_id)
            elif cur_level == level:
                for node in attribute.node_list:
                    value_id_to_vn[value_id] = node
                    vn_to_value_id[node] = value_id
                    value_id += 1
            return value_id

        def node_values(attribute, res):
            res.extend(attribute.value_list)
            for node in attribute.node_list:
                node_values(node, res)

        self.value_id_to_vn = [0]*self.attr_num
        self.value_to_value_id = [0]*self.attr_num
        self.subattr_value_id_to_value = [0]*self.attr_num
        self.subattr_value_to_value_id = [0]*self.attr_num
        
        # attr is column number, subattrs are sorted by their attrs
        self.attr_to_subattr = [-1]*self.attr_num
        self.subattr_num = self.attr_num
        for attr in range(self.attr_num):
            if self.attr_to_level[attr] == self.max_level[attr]:
                continue

            self.attr_to_subattr[attr] = self.subattr_num
            
            # value id to value/node
            value_id_to_vn = {}
            # value/node to value id
            vn_to_value_id = {}
            value_to_attr_value(self.attr_list[attr], 0, self.attr_to_level[attr],\
                0, value_id_to_vn, vn_to_value_id)
            # value to value id
            value_to_value_id = {}

            self.subattr_value_id_to_value.append({})
            self.subattr_value_to_value_id.append({})

            subattr_size = 0
            # subattr value to value id
            subattr_value_to_value_id = {}
            for item in vn_to_value_id:
                if isinstance(item, Attribute):
                    node = item
                    node_value_id = 0
                    node_id = vn_to_value_id[node]
                    values = []
                    node_values(node, values)
                    subattr_size = max(len(values), subattr_size)

                    # subattr value id to value
                    subattr_value_id_to_value = {}

                    for value in values:
                        value_to_value_id[value] = node_id

                        subattr_value_id_to_value[node_value_id] = value
                        subattr_value_to_value_id[value] = node_value_id
                        node_value_id += 1

                    self.subattr_value_id_to_value[self.subattr_num][node] = subattr_value_id_to_value
                else:
                    value_to_value_id[item] = vn_to_value_id[item]

            self.subattr_value_to_value_id[self.subattr_num] = subattr_value_to_value_id

            self.domain.dict[self.subattr_num] = {'domain': subattr_size}

            self.value_id_to_vn[attr] = value_id_to_vn
            self.value_to_value_id[attr] = value_to_value_id

            self.subattr_num += 1


        print('  subattr: ', self.attr_to_subattr)
        self.domain = Domain(self.domain.dict, list(range(self.subattr_num)))

        print('  attrs: ', self.domain.attr_list)
        print('  shape: ', self.domain.shape)

    def map_data_with_hierarchy(self):
        data_list = []
        for line in self.data:
            new_line = list(line) + [0]*(self.subattr_num-self.attr_num)
            for attr in range(self.attr_num):
                if self.attr_to_level[attr] == self.max_level[attr]:
                    continue
                value_id = self.value_to_value_id[attr][line[attr]]
                new_line[attr] = value_id
                
                subattr = self.attr_to_subattr[attr]
                item = self.value_id_to_vn[attr][value_id]
                if isinstance(item, Attribute):
                    new_line[subattr] = self.subattr_value_to_value_id[subattr][line[attr]]
                else:
                    new_line[subattr] = 0

            data_list.append(new_line)
        self.data = np.array(data_list)

    def map_data_with_hierarchy_back(self, np_data):
        print('map synthetic data back to original domain')
        data_list = []
        for line in np_data:
            new_line = list(line)[:self.attr_num]
            for attr in range(self.attr_num):
                if self.attr_to_level[attr] == self.max_level[attr]:
                    continue
                item = self.value_id_to_vn[attr][line[attr]]
                if isinstance(item, Attribute):
                    subattr = self.attr_to_subattr[attr]
                    new_line[attr] = self.subattr_value_id_to_value[subattr][item][line[subattr]]
                else:
                    new_line[attr] = item
            data_list.append(new_line)
        return data_list

    def print_measure_count(self, measure_set):
        measure_length = [0]
        for measure in measure_set:
            if len(measure) > len(measure_length)-1:
                measure_length.extend([0] * (len(measure)-len(measure_length)+1))
            measure_length[len(measure)] += 1
        print('    measure count: ', measure_length[1:])

    # calculate noise scale according to hyperparameters and Gaussain mechanism
    def generte_noise_scale(self):

        if self.config['use_exp_mech'] > 0:
            total_privacy_budget = tools.privacy_budget(self.config['epsilon']*(1-self.config['use_exp_mech']))
        else:
            total_privacy_budget = tools.privacy_budget(self.config['epsilon'])
        privacy_budget2 = self.config['beta2'] * total_privacy_budget
        
        privacy_budget4 = self.config['beta4'] * total_privacy_budget

        if len(self.measure_set) == 0:
            self.de_mesaure_num = int(1.5 * self.attr_num)
            self.marginal_noise = math.sqrt(self.de_mesaure_num / privacy_budget4)
        else:
            if self.config['noise_type'] == 'normal':
                self.marginal_distribution_budget = (1-self.config['beta1']-self.config['beta3']) * total_privacy_budget
                self.de_mesaure_num = int(self.config['t']*self.attr_num)
                self.marginal_noise = math.sqrt((len(self.measure_set) + self.de_mesaure_num) / self.marginal_distribution_budget)
            elif self.config['noise_type'] == 'Laplace':
                # this is for PGM + PrivBayes, beta2 is valid
                self.marginal_noise = len(self.measure_set) / self.config['epsilon'] / self.config['beta2']
                self.de_mesaure_num = -1
            # self.de_mesaure_num = int(privacy_budget4 * self.marginal_noise * self.marginal_noise)

         # For Gaussian distribution, MAD = sigma * (2/math.pi) ** 0.5
        self.max_measure_dom_2way = self.noisy_data_num / (self.marginal_noise * (2/math.pi) ** 0.5) / self.config['theta1']
        self.max_measure_dom_high_way = self.noisy_data_num / (self.marginal_noise * (2/math.pi) ** 0.5) / self.config['theta2']

        if self.config['structure_entropy']:
            self.marginal_noise = 0

        print('  marginal noise:           ', self.marginal_noise)
        print('  max 2way measure dom:     ', self.max_measure_dom_2way)
        print('  max high way measure dom: ', self.max_measure_dom_high_way)
        print('  max de measure num        ', self.de_mesaure_num)

    def generte_noise_scale_de(self):
        if self.config['init_measure'] == 3:
            self.config['ed_step_num'] = min(int(self.attr_num), 12)
        else:
            self.config['ed_step_num'] = min(int(self.attr_num / 2), 8)
        print('ed_step_num: {}'.format(self.config['ed_step_num']))

        total_privacy_budget = tools.privacy_budget(self.config['epsilon'])
        if self.config['use_exp_mech'] < 0:
            self.privacy_budget3 = self.config['beta3'] * total_privacy_budget


    def get_theoretic_loss(self):
        loss = 0
        for measure in self.measure_set:
            loss += self.domain.project(measure).size() * self.marginal_noise * self.marginal_noise
        self.theoretic_loss = 1/2 * loss
        return self.theoretic_loss

    def print_total_variation_distance(self):
        # print('  print all measure')
        # self.print_measure(self.all_measure_set)
        print('  print measure set')
        self.print_measure(self.measure_set)

    def dump_entropy(self, path, entropy, i):
        entropy_log = {}
        if os.path.exists(path):
            with open(path, 'r') as in_file:
                entropy_log = json.load(in_file)
        if self.config['data'] not in entropy_log:
            entropy_log[self.config['data']] = {}
        epsilon = str(self.config['epsilon'])
        if epsilon not in entropy_log[self.config['data']]:
            entropy_log[self.config['data']][epsilon] = {}
        entropy_log[self.config['data']][epsilon][str(i)] = entropy

        with open(path, 'w') as out_file:
            json.dump(entropy_log, out_file)

    # main process
    def entropy_descent(self):
        self.consider_measure_list = self.generate_measure_set()
        print('consider measure set:')
        self.print_measure_count([item[0] for item in self.consider_measure_list])

        consider_measure_list = self.consider_measure_list.copy()
        # self.print_total_variation_distance()

        print('theoretic loss: {:.4e}'.format(self.get_theoretic_loss()))
        if self.config['ed_step_num'] == 0:
            self.config['convergence_ratio'] = self.config['final_convergence_ratio']
        if len(self.measure_set) != 0:
            self.estimate_parameters()
        entropy = self.get_entropy()
        print('entropy: {}'.format(entropy))
        if self.config['structure_entropy']:
            self.dump_entropy('./entropy_log.json', entropy, 'init')
        # self.print_total_variation_distance()

        if self.config['ed_step_num'] == 0:
            return

        measure_num_per_iter = int(self.de_mesaure_num/self.config['ed_step_num'])
        measure_num_list = [measure_num_per_iter] * self.config['ed_step_num']
        for i in range(self.de_mesaure_num - measure_num_per_iter * self.config['ed_step_num']):
            measure_num_list[i] += 1

        print('measure num list', measure_num_list)
        self.ed_step_num = self.config['ed_step_num']
        for i in range(self.ed_step_num-1, -1, -1):
            if measure_num_list[i] == 0:
                self.ed_step_num = i
            else:
                break
        for i in range(self.ed_step_num):
            
            print('entropy descent step: {}/{}'.format(i, self.ed_step_num))
            print('  consider measure set:')
            self.print_measure_count([item[0] for item in consider_measure_list])

            min_it = 0
            if i == self.ed_step_num - 1:
                self.config['convergence_ratio'] = self.config['final_convergence_ratio']
                min_it = self.average_it/i * 3
            if self.config['use_exp_mech'] > 0:
                new_measure_list, new_consider = self.select_measure_exp(consider_measure_list, measure_num_list[i])
            else:
                new_measure_list, new_consider = self.select_measure(consider_measure_list, measure_num_list[i])
            consider_measure_list = new_consider
            print('  measure num: {}'.format(len(self.measure_set)))

            if len(new_measure_list) == 0 and i != self.ed_step_num - 1:
                continue

            self.add_potentials(new_measure_list)
            self.print_measure_count(self.measure_set)
            print('  theoretic loss: {:.4e}'.format(self.get_theoretic_loss()))

            self.estimate_parameters(min_it)
            # self.print_total_variation_distance()
            entropy = self.get_entropy()
            print('entropy: {}'.format(entropy))
            # self.print_measure(self.sampled_measure_list)
            if self.config['structure_entropy']:
                self.dump_entropy('./entropy_log.json', entropy, i)

        with open('./temp/log.txt', 'w') as out_file:
            out_file.write(str(self.measure_set))
        # MarkovRandomField.save_model(self, './temp/model.mrf')

    # add new potentials to the graphical model
    def add_potentials(self, new_measure):
        if self.config['enable_attribute_hierarchy']:
            measure_dict, measure_set = self.query_marginal_with_hierarchy(new_measure)
            for measure in measure_set:
                self.measure[measure] = measure_dict[measure]
                self.measure_set.add(measure)

                for clique in self.maximal_cliques:
                    if set(measure) <= set(clique):
                        self.clique_to_measure[clique].append(measure)
                        self.measure_to_clique[measure] = clique
                        break
        else:
            for measure in new_measure:
                measure = tuple(sorted(measure))
                measure_marginal = tools.dp_marginal_list(self.data, self.domain, [measure], self.marginal_noise, noise_type=self.config['noise_type'])
                temp = measure_marginal[measure]
                if self.gpu:
                    self.measure[measure] = Factor(temp.domain, temp.values, cp)
                else:
                    self.measure[measure] = temp
                self.measure_set.add(measure)

                for clique in self.maximal_cliques:
                    if set(measure) <= set(clique):
                        self.clique_to_measure[clique].append(measure)
                        self.measure_to_clique[measure] = clique
                        break
    
    def print_dom(self, measure_list):
        for measure in measure_list:
            print(measure, self.domain.project(measure).size())

    # generate all the possible measures
    def generate_measure_set(self):
        clique_to_measure = {}
        for clique in self.maximal_cliques:
            clique_to_measure[clique] = []
            print('  generate measure for clique {}, size: {:.2e}'.format(clique, \
                self.domain.project(clique).size()))
            temp_clique = [attr for attr in clique if attr < self.attr_num]
            for length in range(1, self.max_measure_attr_num+1):
                if length >= len(clique):
                    break
                for measure in itertools.combinations(temp_clique, length):
                    measure = tuple(sorted(measure))
                    if length == 1:
                        # Typically, single attribute measure should have no dom constraion as it provides
                        # the basic description for that dom. However, if the dom of the attribute is 
                        # very very large compared to theta-useful. The noise of the single attribute measure
                        # could have a negative effect to the overall parameters. You may want to delete the
                        # single attribute measure in this case. In our experiment, there is no such phenomena.
                        clique_to_measure[clique].append(measure)
                    elif length == 2:
                        if self.domain.project(measure).size() < self.max_measure_dom_2way:
                            clique_to_measure[clique].append(measure)
                    elif self.domain.project(measure).size() < self.max_measure_dom_high_way:
                        clique_to_measure[clique].append(measure)
        

        # normalize to avoid only consider measures in large cliques
        # different clique may generate same measure, which is okay since it is important for both cliques.
        measure_list = []
        for clique in clique_to_measure:
            if len(clique_to_measure[clique]) == 0:
                continue
            weight = len(clique)**2/len(clique_to_measure[clique])
            for measure in clique_to_measure[clique]:
                measure_list.append(tuple([measure, weight]))

        measure_dict = {}
        for item in measure_list:
            if item[0] in measure_dict:
                measure_dict[item[0]] += item[1]
            else:
                measure_dict[item[0]] = item[1]

        measure_list = list(measure_dict.items())
        temp_list = [item[0] for item in measure_list]
        random.shuffle(temp_list)
        self.sampled_measure_list = temp_list[:300]
        # print('measure_list', len(measure_list))
        # print(measure_list)

        # measure_list = [(6, 9), (11,), (13,), (5, 8), (3, 14), (6, 14), (4, 9), (5, 9, 14), (1, 14), (7, 9, 14), (1, 5), (4, 14), (10,), (12,), (0,), (2,), (5, 7), (1, 8), (1, 7)]
        # self.print_dom(measure_list)
        # measure_list = [(item, 1) for item in measure_list]

        if len(measure_list) < self.de_mesaure_num:
            if self.config['use_exp_mech'] > 0:
                total_privacy_budget = tools.privacy_budget(self.config['epsilon']*(1-self.config['use_exp_mech']))
            else:
                total_privacy_budget = tools.privacy_budget(self.config['epsilon'])
            self.de_mesaure_num = len(measure_list)
            self.marginal_noise = math.sqrt((len(self.measure_set)+self.de_mesaure_num) / self.marginal_distribution_budget)
            print('  all the measures can be selected')
            print('  marginal noise:           ', self.marginal_noise)
            print('  max de measure num        ', self.de_mesaure_num)

        # assign measures to cliques
        for measure, weight in measure_list:
            for clique in self.maximal_cliques:
                if set(measure) <= set(clique):
                    self.clique_to_measure[clique].append(measure)
                    self.measure_to_clique[measure] = clique
                    break

        return measure_list

    def print_measure(self, print_measure):
        marginal_dict, partition_func = self.cal_marginal_dict(self.potential, print_measure, to_cpu=True)
        average = 0
        for measure in print_measure:
            bins = self.domain.project(measure).edge()
            histogram, _ = np.histogramdd(self.data[:, measure], bins=bins)

            value = marginal_dict[measure].values
            total_variation_distance = np.sum(np.abs(value - histogram)) / 2 / self.noisy_data_num
            average += total_variation_distance
        measure_num = len(print_measure)
        print('  average total variation distance: {:.8f}'.format(average/measure_num))

    # select measures that have large norm
    def select_measure(self, measure_list, measure_num):
        print('  ', len(measure_list), measure_num)
        if measure_num <= 0:
            return []
        measure_list = list(measure_list.copy())
        if measure_num > len(measure_list):
            return [item[0] for item in measure_list]

        # limit the number of queries to ensure privacy budget is enough
        if len(measure_list) > self.config['norm_query_number']:
            weights = [item[1] for item in measure_list]
            query_measure_list = random.choices(measure_list, weights=weights, k=self.config['norm_query_number'])
            query_measure_list = [item[0] for item in query_measure_list]
        else:
            query_measure_list = [item[0] for item in measure_list]
        query_measure_list = list(set(query_measure_list))

        budget = self.privacy_budget3/self.ed_step_num
        self.de_norm1_noise = math.sqrt(len(query_measure_list)/budget)
        print('  marginal norm noise:', self.de_norm1_noise)

        # query 1 norm
        query_result_list = []
        marginal_dict, partition_func = self.cal_marginal_dict(self.potential, query_measure_list, to_cpu=True)
        for measure in query_measure_list:
            dist, noisy_dist = tools.dp_1norm(self.data, self.domain, measure, marginal_dict[measure], self.de_norm1_noise, to_cpu=True)
            # TVD (1 norm) of marginal is at least this value. Deduct it to compare marginals of different sizes fairly
            # However, it is not emperically better in adult dataset. There should be better ways to compare marginal with different size
            # 1. inherent TVD given by noise. 2. TVD tends to be large if the size of noisy marginal is large.
            # dist -= self.marginal_noise * self.domain.project(measure).size() / 2 / self.noisy_data_num
            query_result_list.append([measure, noisy_dist, dist])
        query_result_list.sort(key=lambda x: x[1], reverse=True)

        # sort and find measures with maximum 1 norm
        measure_num = min(measure_num, len(query_result_list))
        result_list = query_result_list[: measure_num]
        print('  new selected measure list')
        for i in range(measure_num):
            print('   ', result_list[i])
        result_list = [x[0] for x in result_list]

        print('  consider measure list', len(measure_list))
        # print(measure_list)
        new_measure_list = []
        temp_list = query_measure_list[int(len(query_measure_list)/2):]
        for item in measure_list:
            if item[0] not in result_list:
                if item[0] in temp_list:
                    new_measure_list.append((item[0], item[1]/2))
                else:
                    new_measure_list.append(item)

        # decrease attribute level for measures
        if self.config['enable_attribute_hierarchy']:
            new_result_list = []
            for measure in result_list:
                new_result_list.append(self.get_measure_with_hierarchy(measure))
            result_list = new_result_list
        
        return result_list, new_measure_list

    def select_measure_exp(self, measure_list, measure_num):
        if measure_num <= 0:
            return []
        measure_list = list(measure_list.copy())
        if measure_num > len(measure_list):
            return [item[0] for item in measure_list]

        # limit the number of queries to ensure privacy budget is enough
        if len(measure_list) > self.config['norm_query_number']:
            weights = [item[1] for item in measure_list]
            query_measure_list = random.choices(measure_list, weights=weights, k=self.config['norm_query_number'])
            query_measure_list = [item[0] for item in query_measure_list]
        else:
            query_measure_list = [item[0] for item in measure_list]
        query_measure_list = list(set(query_measure_list))

        # query 1 norm
        marginal_dict, partition_func = self.cal_marginal_dict(self.potential, query_measure_list, to_cpu=True)
        query_measure_list = []
        query_score_list = []
        for measure in marginal_dict:
            dist, noisy_dist = tools.dp_1norm(self.data, self.domain, measure, marginal_dict[measure], 0, to_cpu=True)
            query_measure_list.append(measure)
            query_score_list.append(dist)

        result_list = []
        budget = self.config['epsilon']/self.ed_step_num * self.config['use_exp_mech']/measure_num
        for i in range(measure_num):
            measure, choice = tools.exponential_mechanism(query_measure_list, query_score_list, budget, 1/2/self.data_num)
            result_list.append(measure)
            del query_measure_list[choice]
            del query_score_list[choice]
        
        new_measure_list = []
        for item in measure_list:
            if item[0] not in result_list:
                new_measure_list.append(item)
        
        if self.config['enable_attribute_hierarchy']:
            new_result_list = []
            for measure in result_list:
                new_result_list.append(self.get_measure_with_hierarchy(measure))
            result_list = new_result_list
        
        return result_list, new_measure_list    
    

    def estimate_parameters(self, min_it=0):
        if self.config['estimation_method'] == 'mirror_descent':
            self.mirror_descent(min_it)
        elif self.config['estimation_method'] == 'accelerated_mirror_descent':
            self.accelerated_mirror_descent()
        elif self.config['estimation_method'] == 'dual_averaging':
            self.dual_averaging()
        else:
            print('error: invalid estimation method')
            exit(-1)

    def mirror_descent(self, min_it=0):
        print('mirror descent')
        potential = self.potential.copy()

        mu = None
        alpha = 1.0 /self.noisy_data_num ** 2
        stepsize = lambda t: 2.0*alpha

        mu, partition_func = self.cal_marginal_dict(potential, self.measure_set)
        ans = Potential.l2_marginal_loss(mu, self.measure)

        for it in range(self.iter_num):
            start_time = time.time()
            omega, nu = potential, mu
            curr_loss, gradient = ans
            alpha = stepsize(it)
            expanded_gradient = self.get_expanded_gradient(gradient)
            for i in range(25):
                potential = omega - alpha * expanded_gradient

                mu, partition_func = self.cal_marginal_dict(potential, self.measure_set)
                ans = Potential.l2_marginal_loss(mu, self.measure)
                if curr_loss - ans[0] >= 0.5*alpha*gradient.dot(nu-mu):
                    break
                alpha *= 0.5

            if it % self.config['print_interval'] == 0 or it == self.iter_num-1:
                print('    it: {}/{} loss: {:.4e} time: {:.2f}'.format(it, self.iter_num, curr_loss, time.time()-start_time))
                if curr_loss < self.config['convergence_ratio']*self.theoretic_loss and it > min_it:
                    break

        self.average_it += it

        self.partition_func = partition_func
        self.potential = potential
        self.marginal = mu

    def get_expanded_gradient(self, gradient):
        if self.gpu:
            xp = cp
        else:
            xp = np
        expanded_gradient = Potential({clique: Factor.zeros(\
            self.domain.project(clique), xp) for clique in self.maximal_cliques})
        for marginal in gradient:
            clique = self.measure_to_clique[marginal]
            expanded_gradient[clique] += gradient[marginal]
        return expanded_gradient

    # get entropy of current model
    def get_entropy(self):
        marginal, partition_func = self.belief_propagation(self.potential) 
        ans = - self.potential.dot(1/self.noisy_data_num * marginal) + partition_func
        return ans.item()

    def cal_marginal_dict(self, potential, measure_set, to_cpu=False):
        maximal_clique_marginal, partition_func = self.belief_propagation(potential)
        if to_cpu:
            for clique in maximal_clique_marginal:
                maximal_clique_marginal[clique] = maximal_clique_marginal[clique].to_cpu()

        marginal_dict = {}
        for marginal in measure_set:
            clique_factor = maximal_clique_marginal[self.measure_to_clique[marginal]]
            marginal_fact = clique_factor.project(marginal)
            marginal_dict[marginal] = marginal_fact
        return Potential(marginal_dict), partition_func

    def accelerated_mirror_descent(self):
        print('accelerated mirror descent')
        potential = self.potential.copy()
        L = self.get_Lipschitz_constant()
        print('Lipschitz constant: {:.4e}'.format(L))
        x_t, partition_func = self.cal_marginal_dict(potential, self.measure_set)
        y_t = x_t.copy()
        z_t = x_t.copy()

        alpha = lambda t: 2/(t+2)
        it = 0
        while True:
            it += 1
            start_time = time.time()

            y_t = (1 - alpha(it)) * x_t + alpha(it)*z_t
            loss, gradient = Potential.l2_marginal_loss(y_t, self.measure)

            for measure in gradient:
                clique = self.measure_to_clique[measure]
                potential[clique] = potential[clique] - 1/alpha(it)/L * (1/self.noisy_data_num) * gradient[measure]

            for clique in self.maximal_cliques:
                potential[clique].values[potential[clique].values < 0] = 0
            z_t, partition_func = self.cal_marginal_dict(potential, self.measure_set)
            x_t = (1-alpha(it)) * x_t + alpha(it) * z_t

            if it % self.config['print_interval'] == 0 or it == self.iter_num-1:
                print('    it: {}/{} loss: {:.4e} time: {:.2f}'.format(it, self.iter_num, loss, time.time()-start_time))
                if loss < self.config['convergence_ratio']*self.theoretic_loss:
                    break

            if it > self.iter_num:
                break
    
        self.partition_func = partition_func
        self.potential = potential
        self.marginal = z_t

    def dual_averaging(self):
        print('dual averaging')
        # Dual averaging requires starting from the point with the maximimal entropy
        if self.gpu:
            xp = cp
        else:
            xp = np
        potential = Potential({clique: Factor.zeros(self.domain.project(clique), xp) for clique in self.maximal_cliques})
        L = self.get_Lipschitz_constant()
        print('Lipschitz constant: {:.4e}'.format(L))
        mu, partition_func = self.cal_marginal_dict(potential, self.measure_set)
        v = mu
        g = Potential({clique: Factor.zeros(self.domain.project(clique), xp) for clique in self.maximal_cliques})
        it = 0
        while True:
            it += 1
            start_time = time.time()

            c = 2/(1+it)
            w = (1-c)*mu + c*v
            loss, gradient = Potential.l2_marginal_loss(w, self.measure)
            
            g = (1-c)*g
            for measure in gradient:
                clique = self.measure_to_clique[measure]
                g[clique] = g[clique] + c*gradient[measure]
            potential = -it*(it+1) / (4*L*self.noisy_data_num) * g

            v, partition_func = self.cal_marginal_dict(potential, self.measure_set)
            mu = (1-c)*mu + c*v

            if it % self.config['print_interval'] == 0 or it == self.iter_num-1:
                print('    it: {}/{} loss: {:.4e} time: {:.2f}'.format(it, self.iter_num, loss, time.time()-start_time))
                if loss < self.config['convergence_ratio']*self.theoretic_loss:
                    break

            if it > self.iter_num:
                break

        self.partition_func = partition_func
        self.potential = potential
        self.marginal = mu

    # calculate margianls of maximal cliques
    def belief_propagation(self, potential):
        belief = Potential({clique: potential[clique].copy() for clique in self.maximal_cliques})

        sent_message = dict()
        for clique1, clique2 in self.message_order:
            separator = set(clique1) & set(clique2)
            if (clique2, clique1) in sent_message:
                message = belief[clique1] - sent_message[(clique2, clique1)]
            else:
                message = belief[clique1]
            message = message.logsumexp(separator)
            belief[clique2] += message

            sent_message[(clique1, clique2)] = message

        partition_func = belief[self.maximal_cliques[0]].logsumexp()
        for clique in self.maximal_cliques:
            belief[clique] += np.log(self.noisy_data_num) - partition_func
            belief[clique] = belief[clique].exp()

        return belief, partition_func

    def get_Lipschitz_constant(self):
        L = {clique: 0 for clique in self.maximal_cliques}
        for measure in self.measure_set:
            clique = self.measure_to_clique[measure]
            L[clique] += self.domain.project(clique).size() / self.domain.project(measure).size() / len(self.measure_set)
        self.L = max(L.values())
        return self.L

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as out_file:
            pickle.dump(model, out_file)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as out_file:
            return pickle.load(out_file)

    # generate synthetic data according to potentials
    def synthetic_data(self, path):
        data = np.zeros((self.noisy_data_num, len(self.domain)), dtype=int)
        self.df = pd.DataFrame(data, columns=self.domain.attr_list)
        # belief propagation to get clique marginals and
        # generate data conditioned on separators
        clique_marginal, partition_func = self.belief_propagation(self.potential)
        finished_attr = set()
        separator = set()
        if len(self.maximal_cliques) == 1:
            cond_attr = []
            clique = self.maximal_cliques[0]
            for attr in clique:
                print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))
                self.pandas_generate_cond_column_data(clique_marginal[clique], cond_attr, attr)
                finished_attr.add(attr)
                cond_attr.append(attr)
        else:
            for start, clique in nx.dfs_edges(self.junction_tree):
                if len(finished_attr) == 0:
                    cond_attr = []
                    for attr in start:
                        print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))
                        self.pandas_generate_cond_column_data(clique_marginal[start], cond_attr, attr)
                        finished_attr.add(attr)
                        cond_attr.append(attr)

                separator = set(start) & set(clique)
                print('start: {}, clique: {}, sep: {}'.format(start, clique, separator))
                cond_attr = list(separator)
                for attr in clique:
                    if attr not in finished_attr:
                        print('  cond_attr: {}, attr: {} {}/{}'.format(cond_attr, attr, len(finished_attr), len(self.domain.attr_list)))
                        self.pandas_generate_cond_column_data(clique_marginal[clique], cond_attr, attr)
                        finished_attr.add(attr)
                        cond_attr.append(attr)

        if self.config['enable_attribute_hierarchy']:
            data_list = self.map_data_with_hierarchy_back(self.df.to_numpy())
        else:
            data_list = list(self.df.to_numpy())
        tools.write_csv(data_list, list(range(self.attr_num)), path)
        return data_list

    # generate a column according to marginal distribution and conditions using pandas
    def pandas_generate_cond_column_data(self, clique_factor, cond, target):
        clique_factor = clique_factor.moveaxis(self.domain.attr_list)
        if len(cond) == 0:
            prob = clique_factor.project(target).values
            self.df.loc[:, target] = tools.generate_column_data(prob, self.noisy_data_num)
        else:
            marginal_value = clique_factor.project(cond + [target])

            attr_list = marginal_value.domain.attr_list.copy()
            attr_list.remove(target)
            cond = attr_list.copy()
            attr_list.append(target)

            marginal_value = marginal_value.moveaxis(attr_list).values

            if self.config['enable_attribute_hierarchy']:
                attrs = [attr for attr in attr_list if attr < self.attr_num and self.attr_to_subattr[attr] in attr_list]
                self.set_zero_for_hierarchy(marginal_value, attr_list, attrs)

            def foo(group):
                idx = group.name
                vals = tools.generate_column_data(marginal_value[idx], group.shape[0])
                group[target] = vals
                return group

            self.df = self.df.groupby(list(cond)).apply(foo)
