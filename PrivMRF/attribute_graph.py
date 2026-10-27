from .utils import tools
from .attribute_hierarchy import expanded_size
from .domain import Domain
import networkx as nx
import numpy as np
import random
import itertools
import math
from networkx.readwrite import json_graph
import json
import os

class Status:
    def __init__(self, graph, adj):
        self.graph = graph.copy()
        self.adj = np.copy(adj)

    def get_neighbor_status(self, attr1, attr2, weight):
        neighbor_status = Status(self.graph, self.adj)
        if neighbor_status.adj[attr1, attr2] > 0:
            neighbor_status.adj[attr1, attr2] = 0
            neighbor_status.adj[attr2, attr1] = 0
            neighbor_status.graph.remove_edge(attr1, attr2)
            return neighbor_status
        elif neighbor_status.adj[attr1, attr2] == 0:
            neighbor_status.adj[attr1, attr2] = weight
            neighbor_status.adj[attr2, attr1] = weight
            neighbor_status.graph.add_edge(attr1, attr2)
            return neighbor_status
        return None

class AttributeGraph:
    def __init__(self, data, domain, attr_list, config):
        self.data = data.copy()
        self.domain = domain
        self.attr_list = attr_list
        self.config = config

        self.attr_num = len(domain)
        self.privacy_budget = config['beta1'] * config['privacy_budget']

        self.TVD_map = {}

        # noisy number of records, which costs beta_n of the privacy budget
        data_num_noise = 1 / math.sqrt(config['beta_n'] * config['privacy_budget'])
        self.data_num = int(len(self.data) + np.random.normal(scale=data_num_noise))

        self.config['search_iter_num'] = int(self.attr_num*(self.attr_num-1))

        max_edge_num = self.attr_num * (self.attr_num - 1)/2

        sensitivity = tools.TVD_sensitivity(self.data_num)
        self.TVD_noise = math.sqrt(max_edge_num * sensitivity * sensitivity / self.privacy_budget)

        self.max_measure_dom_2way = 0
        self.max_measure_dom_high_way = 0
        if self.config['init_measure'] == 3:
            self.config['init_measure_num'] = 0
        else:
            self.config['init_measure_num'] = self.attr_num

        estimated_noise_scale = 'nan'
        if self.config['init_measure'] != 3:
            budget = (1-self.config['beta1']-self.config['beta3']-self.config['beta_n']) * config['privacy_budget']
            estimated_noise_scale = math.sqrt((self.attr_num + self.config['t']*self.attr_num) / budget)
            self.max_measure_dom_2way = self.data_num / estimated_noise_scale / config['theta1']
            self.max_measure_dom_high_way = self.data_num / estimated_noise_scale / config['theta2']

        self.min_score = -1e8

        self.max_level = [max(self.attr_list[attr].level_to_size.keys()) for attr in range(self.attr_num)]

        # levels of attributes in their hierarchies, None if attribute hierarchy is disabled
        self.attr_to_level = None
        if self.config['enable_attribute_hierarchy']:
            self.attr_to_level = {attr: self.max_level[attr] for attr in range(self.attr_num)}
            if self.config['init_measure'] != 3 and self.config['attr_level'] is not None:
                self.attr_to_level.update(self.config['attr_level'])
        self.update_size_domain()

        # the best measure for each attr to avoid worst cases
        self.attr_measure = {}

        print('privacy budget:           ', self.privacy_budget)
        print('estimated noise scale:    ', estimated_noise_scale)
        print('max 2way measure dom:     ', self.max_measure_dom_2way)
        print('max high way measure dom: ', self.max_measure_dom_high_way)
        print('max edge number:          ', max_edge_num)
        print('TVD_noise:                ', self.TVD_noise)
        print('data num:                 ', self.data_num)

    def construct_model(self):
        print('construct attribute graph')
        self.graph = self.local_search()

        if self.config['enable_attribute_hierarchy']:
            for attr in range(self.attr_num):
                if self.max_level[attr] > 0:
                    print('attr: {} max_level: {}'.format(attr, self.max_level[attr]))

        measure_list = []
        if self.config['init_measure'] == 0:
            measure_list = self.construct_inner_Bayesian_network()
        elif self.config['init_measure'] == 1:
            measure_list = self.get_all_n_way_measure(2)
        elif self.config['init_measure'] == 2:
            measure_list = list(tuple(sorted(clique)) for clique in nx.find_cliques(self.graph))
        elif self.config['init_measure'] == 3:
            self.measure_list = []
            return self.graph, self.measure_list, self.attr_list, self.attr_to_level
        else:
            raise ValueError('invalid init_measure: {}'.format(self.config['init_measure']))

        # add most valuable measures for attrs
        self.measure_list = []
        for attr in self.attr_measure:
            self.measure_list.append(self.attr_measure[attr][0])
        print('attr measure',  self.measure_list)
        self.measure_list = \
            list(tools.deduplicate_measure_set(tuple(sorted(measure)) \
                for measure in self.measure_list))

        # determine the level of attribute hierarchy
        if self.config['enable_attribute_hierarchy']:
            for measure in self.measure_list:
                if len(measure) == 2:
                    for i in range(3):
                        if tools.measure_level_size(measure, self.attr_list, self.attr_to_level) > self.max_measure_dom_2way:
                            attr_to_level = self.attr_to_level.copy()
                            tools.improve_level(measure, self.attr_list, self.attr_to_level, self.config['max_level_gap'])
                            self.update_size_domain()
                            # keep the level if the subattribute would make the model too large
                            if self.model_size(self.graph) is None:
                                self.attr_to_level = attr_to_level
                                self.update_size_domain()
                                break

            for attr in self.attr_to_level:
                if self.max_level[attr] > 0:
                    print('  attr: {}, level: {} max_level: {}'.format(attr, self.attr_to_level[attr], self.max_level[attr]))

        attr_flag = [0] * self.attr_num
        for measure in self.measure_list:
            for attr in measure:
                attr_flag[attr] += 1
        for attr in range(self.attr_num):
            if attr_flag[attr] == 0:
                self.measure_list.append(tuple([attr]))
                print('single attr measure:', attr)

        # if there remains space for other measures, add them
        measure_list = \
            list(tools.deduplicate_measure_set(tuple(sorted(measure)) \
                for measure in measure_list))
        random.shuffle(measure_list)
        for measure in measure_list:
            if len(self.measure_list) >= self.config['init_measure_num']:
                break
            if not tools.contain_measure(self.measure_list, measure):
                self.measure_list.append(measure)

        if self.config['log_dir'] is not None:
            data = json_graph.node_link_data(self.graph, edges='edges')
            with open(os.path.join(self.config['log_dir'], self.config['exp_name']+'_graph.json'), 'w') as out_file:
                json.dump(data, out_file)
            with open(os.path.join(self.config['log_dir'], self.config['exp_name']+'_measure.json'), 'w') as out_file:
                json.dump(self.measure_list, out_file)

        print('model size with attribute hierarchy:', self.model_size(self.graph))
        return self.graph, self.measure_list, self.attr_list, self.attr_to_level

    def get_all_n_way_measure(self, n):
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for clique in self.maximal_cliques:
            for measure in itertools.combinations(clique, n):
                measure_list.append(measure)
        return measure_list

    # randomly enumerate a next edge to find a graph that minimize KL divergence and get measures
    def local_search(self):
        start_G = nx.Graph()
        start_G.add_nodes_from(list(range(self.attr_num)))
        start_adj = np.zeros(shape=(self.attr_num, self.attr_num), dtype=float)

        current_status = Status(start_G, start_adj)

        temp1 = 0
        for attr1 in range(self.attr_num):
            for attr2 in range(attr1+1, self.attr_num):
                temp1 += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr1, attr2], self.TVD_noise)[0]
        print('average TVD and noise:', temp1/(self.attr_num*(self.attr_num-1)/2), self.TVD_noise)

        local_count = 0
        search_iter_num = self.config['search_iter_num']
        for i in range(search_iter_num):

            # generate edge list
            best_score = self.min_score
            best_status = None
            edge_list = []
            for attr1 in range(self.attr_num):
                for attr2 in range(attr1+1, self.attr_num):
                    if current_status.adj[attr1][attr2] == 0:
                        edge_list.append((attr1, attr2))
            random.shuffle(edge_list)

            for attr1, attr2 in edge_list:
                status = current_status.get_neighbor_status(attr1, attr2, 1)
                status_score, noisy_TVD, size = self.pairwise_score_TVD(status.graph)
                if status_score > best_score:
                    best_score = status_score
                    best_status = status
                    best_noisy_TVD = noisy_TVD
                    current_size = size

            if best_status == None:
                print('  found local minimum')
                local_count += 1
                if local_count >= 3:
                    break
                continue
            local_count = 0
            current_status = best_status
            current_score = best_score

            print('  iter: {}/{} score: {:.2f} size: {:.2e}, edge_num: {} noisy_TVD: {:.2f}'\
                .format(i, search_iter_num, current_score, current_size, \
                current_status.graph.number_of_edges(), best_noisy_TVD))

        graph = current_status.graph
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        if self.config['log_dir'] is not None:
            tools.print_graph(graph, os.path.join(self.config['log_dir'], self.config['exp_name']+'_graph.png'))

        return graph

    def construct_inner_Bayesian_network(self):
        print('construct Bayesian Network for maximal cliques')
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for i in range(len(self.maximal_cliques)):
            clique = self.maximal_cliques[i]
            print('  {}, {}/{}'.format(clique, i, len(self.maximal_cliques)))
            measure_list.extend(self.greedy_Bayes(clique))

        if self.config['supplement_2way']:
            for clique in self.maximal_cliques:
                for edge in itertools.combinations(clique, 2):
                    measure_list.append(edge)

        measure_list = list(set(measure_list))
        return measure_list

    def maximal_parents(self, parents_set, dom):
        if dom < 1:
            return set()
        if len(parents_set) < 1:
            return set([tuple(),])

        parents_set = parents_set.copy()
        attr = parents_set.pop()
        res1 = self.maximal_parents(parents_set, dom)

        # If there exists a high level subattr satisfying dom limitation
        # It should be considered as levels don't influence the scores of parents
        if self.config['enable_attribute_hierarchy']:
            level = max(self.max_level[attr]-2, 0)
        else:
            level = max(self.max_level[attr], 0)

        current_attr_size = self.attr_list[attr].level_to_size[level]
        res2 = self.maximal_parents(parents_set, dom/current_attr_size)
        for ps in res2:
            if ps in res1:
                res1.remove(ps)
            temp = list(ps)
            temp.append(attr)
            res1.add(tuple(sorted(temp)))

        return res1

    def select_parents(self, remaining_attributes, parents_set):
        attr_parents = []
        for attr in remaining_attributes:
            dom_limit = self.max_measure_dom_high_way/self.domain.project([attr]).size()
            attr_parents.extend([(attr, parent) for parent in self.maximal_parents(parents_set, dom_limit)])

        best_attr = None
        best_parent = []
        best_score = self.min_score
        for ap in attr_parents:
            score = self.attr_parents_score(ap[0], list(ap[1]))
            if score > best_score:
                best_score = score
                best_attr = ap[0]
                best_parent = ap[1]

        if best_attr == None:
            best_attr = random.choice(remaining_attributes)

        marginal = [best_attr]
        marginal.extend(best_parent)
        marginal = tuple(sorted(marginal))

        return (marginal, best_attr), best_score

    # use greedy bayes to find measures
    def greedy_Bayes(self, clique):
        remaining_attributes = list(clique.copy())
        best_attr = random.choice(remaining_attributes)

        remaining_attributes.remove(best_attr)
        parents_set = [best_attr]

        measure_list = []
        while len(remaining_attributes) != 0:
            marginal_item, score = self.select_parents(remaining_attributes, parents_set)
            best_attr = marginal_item[1]
            best_parents = marginal_item[0]

            if len(marginal_item[0]) > 1:
                print('    {}<={} score: {}'.format(best_attr, best_parents, score))
                best_parents = tuple(sorted(best_parents))
                measure_list.append(best_parents)

                if best_attr not in self.attr_measure or self.attr_measure[best_attr][1] < score:
                    self.attr_measure[best_attr] = [best_parents, score]

                remaining_attributes.remove(best_attr)
                parents_set.append(best_attr)
            else:
                print('unable to construct Bayes network under dom limitation', clique)
                break

        return measure_list


    # TVD correlation + correlation-based feature selector
    # ref: Correlation-based Feature Selection for Machine Learning
    def attr_parents_score(self, attr, parents):
        parents = list(parents)
        # If the measure constructed by the only parents of one attribute is too large,
        # it should be add to the model regardless of its size as it provide basic correlation we can get
        # It will aslo be used to determine attribute hierarchy if possible
        # however, if a measure is too large, the noise will also be very large.
        # It will even affect the entire model
        if len(parents) == 1:
            dom_limit = self.max_measure_dom_2way
            if self.config['enable_attribute_hierarchy']:
                for pa in parents:
                    if self.attr_to_level[pa] > 0:
                        dom_limit = self.max_measure_dom_2way * 5
                if self.attr_to_level[attr] > 0:
                    dom_limit = self.max_measure_dom_2way * 5
        else:
            dom_limit = self.max_measure_dom_high_way
        if self.domain.project(parents).size() * self.domain.dict[attr]['domain'] > dom_limit:
            return self.min_score
        # it will reuse the TVD queried before
        numerator = 0
        for parent in parents:
            numerator += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr, parent], self.TVD_noise)[1]
        denominator = len(parents)
        for parent1, parent2 in itertools.combinations(parents, 2):
            denominator += tools.dp_TVD(self.TVD_map, self.data, self.domain, [parent1, parent2], self.TVD_noise)[1]
        # denominator might be smaller than 0 because of noise. We set it at least 1 as len(parents) >= 1
        if denominator < 1:
            denominator = 1
        return numerator/math.sqrt(denominator)

    # domain for checking the model size, in which an attribute that is not at the finest level of
    # its hierarchy also takes the cells of its subattribute, as in the Markov random field
    def update_size_domain(self):
        size_dict = {}
        for attr in range(self.attr_num):
            if self.attr_to_level is None:
                size_dict[attr] = {'domain': self.domain.dict[attr]['domain']}
            else:
                size_dict[attr] = {'domain': expanded_size(self.attr_list[attr], self.attr_to_level[attr])}
        self.size_domain = Domain(size_dict, list(range(self.attr_num)))

    # junction tree size of a triangulated graph, None if it violates the size limits
    def model_size(self, graph):
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.size_domain.project(clique).size()
            if temp_size > self.config['max_clique_size']:
                return None
            size += temp_size
        if size > self.config['max_parameter_size']:
            return None
        return size

    def pairwise_score_TVD(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)

        size = self.model_size(graph)
        if size is None:
            return self.min_score, 0, 0

        noisy_TVD = 0
        for attr1, attr2 in graph.edges:
            noisy_TVD += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr1, attr2], self.TVD_noise)[1]

        score = noisy_TVD - self.config['size_penalty']*size
        return score, noisy_TVD, size
