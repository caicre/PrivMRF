import csv
import functools
import itertools
import json
import math
import random

import networkx as nx
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..backend import to_numpy
from ..factor import Factor


def read_csv(path, print_info=True):
    if print_info:
        print('    read csv data ' + path)
    data_list = []
    with open(path, 'r', newline='') as in_file:
        reader = csv.reader(in_file)
        headings = next(reader)
        data_list = [line for line in reader]
    return data_list, headings

def write_csv(data_list, headings, path, print_info=True):
    if print_info:
        print('    write csv data ' + path)
    with open(path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(headings)
        for line in data_list:
            writer.writerow(line)

def dp_1norm(data, domain, index_list, marginal, noise):
    bins = domain.project(index_list).edge()
    histogram, _ = np.histogramdd(data[:, index_list], bins=bins)
    value = to_numpy(marginal.values)
    result1 = np.sum(np.abs(value - histogram))
    result2 =  (result1 + np.random.normal(scale=noise)) / 2 /len(data)
    return result1 / 2 /len(data) , result2

def dp_TVD(TVD_map, data, domain, index_list, noise):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in TVD_map:
        domain = domain.project(index_list)
        bins = domain.edge()

        histogram, _= np.histogramdd(data[:, index_list], bins=bins)
        fact1 = Factor(domain, histogram, np)

        temp_domain = domain.project([index_list[0]])
        temp_index_list = temp_domain.attr_list
        histogram, _= np.histogramdd(data[:, temp_index_list], bins=temp_domain.edge())
        fact2 = Factor(temp_domain, histogram, np)

        temp_domain = domain.project([index_list[1]])
        temp_index_list = temp_domain.attr_list
        histogram, _= np.histogramdd(data[:, temp_index_list], bins=temp_domain.edge())
        fact3 = Factor(temp_domain, histogram, np)

        data_num = len(data)
        fact4 = fact2.expand(domain) * fact3.expand(domain) / data_num
        TVD = float(np.sum(np.abs(fact4.values - fact1.values)) / 2 / data_num)
        noisy_TVD = TVD + np.random.normal(scale=noise)

        TVD_map[index_list] = [TVD, noisy_TVD]

    return TVD_map[index_list]

def dp_marginal_list(data, domain, marginal_list, noise, return_factor=True):
    marginal_dict = {}
    for marginal in marginal_list:
        temp_domain = domain.project(marginal)
        histogram, _ = np.histogramdd(data[:, marginal], bins=temp_domain.edge())
        histogram += np.random.normal(scale=noise, size=temp_domain.shape)
        if return_factor:
            fact = Factor(temp_domain, histogram)
            marginal_dict[tuple(marginal)] = fact
        else:
            marginal_dict[tuple(marginal)] = histogram
    return marginal_dict

# sensitivity of TVD
# ref: PrivBayes: Private Data Release via Bayesian Networks
def TVD_sensitivity(n):
    return 2.0/n

def triangulate(graph):
    edges = set()
    G = nx.Graph(graph)

    nodes = sorted(graph.degree(), key=lambda x: x[1])
    for node, degree in nodes:
        local_complete_edges = set(itertools.combinations(G.neighbors(node), 2))
        edges |= local_complete_edges

        G.add_edges_from(local_complete_edges)
        G.remove_node(node)

    triangulated_graph = nx.Graph(graph)
    triangulated_graph.add_edges_from(edges)

    return triangulated_graph

# The privacy budget g of the paper: Gaussian queries with sum_i S(Q_i)^2/sigma_i^2 <= g
# satisfy (epsilon, delta)-DP iff Phi(sqrt(g)/2 - epsilon/sqrt(g))
# - e^epsilon * Phi(-sqrt(g)/2 - epsilon/sqrt(g)) <= delta (Lemma 3, Eq. 14).
# The left side increases with g, so we find the largest g by root finding.
@functools.lru_cache(maxsize=None)
def privacy_budget(epsilon, delta=1e-5):
    def excess_delta(g):
        mu = math.sqrt(g)
        return norm.cdf(mu/2 - epsilon/mu) - math.exp(epsilon)*norm.cdf(-mu/2 - epsilon/mu) - delta

    upper = 1.0
    while excess_delta(upper) < 0:
        upper *= 2
    return brentq(excess_delta, 1e-12, upper, xtol=1e-15, rtol=1e-12)

def contain_measure(measure_list, query_measure):
    for measure in measure_list:
        if len(set(query_measure) - set(measure)) == 0:
            return True
    return False

def deduplicate_measure_set(measure_set):
    measure_list = list(measure_set)
    measure_list.sort(key=lambda x: len(x), reverse=True)
    measure_set = set()
    for i in range(len(measure_list)):
        if not contain_measure(measure_list[:i], measure_list[i]):
            measure_set.add(measure_list[i])
    return measure_set

# generate number data according to prob
def generate_column_data(prob, number):
    prob = to_numpy(prob)
    if np.sum(prob) == 0:
        prob += 1
    prob = prob * number/prob.sum()
    frac, integral = np.modf(prob)
    integral = integral.astype(int)
    round_number = int(number - integral.sum())
    if round_number > 0:
        index = np.random.choice(prob.size, round_number, p=frac/frac.sum())
        unique, unique_counts = np.unique(index,  return_counts=True)
        for i in range(len(unique)):
            integral[unique[i]] += unique_counts[i]
    data = np.repeat(np.arange(prob.size), integral)
    np.random.shuffle(data)
    return data

def print_graph(G, path):
    # draw without pyplot so that no GUI window is opened
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4, 3), dpi=600)
    nx.draw(G, ax=fig.add_subplot(), with_labels=True, edge_color='b', node_color='g', node_size=20, font_size=4, width=0.5)
    fig.savefig(path)

def read_json_domain(domain_path):
    with open(domain_path, 'r') as in_file:
        domain = json.load(in_file)
    domain = {int(attr): domain[attr] for attr in domain}
    return domain

def measure_level_size(measure, attr_list, attr_to_level):
    size = 1
    for attr in measure:
        size *= attr_list[attr].level_to_size[attr_to_level[attr]]
    return size

def improve_level(measure, attr_list, attr_to_level, max_gap, print_info=True):
    choices = []
    # improve level of attribute may reduce the performance of PrivMRF greatly
    # if you find the domain of attribute is too large, consider preprocess your data
    # to split domain properly or improve epsilon
    min_gap = max_gap
    for attr in measure:
        if attr_to_level[attr] <= 0:
            continue
        gap = max(attr_list[attr].level_to_size.keys()) - attr_to_level[attr]
        if gap < min_gap:
            choices = [attr]
            min_gap = gap
        elif gap == min_gap and gap < max_gap:
            choices.append(attr)
    if len(choices) >= 1:
        attr = random.choices(choices)[0]
        attr_to_level[attr] -= 1
        if print_info:
            print('    improve measure: {}, attr: {}'.format(measure, attr))
