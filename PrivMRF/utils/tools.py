import csv
from scipy import stats
from functools import reduce
import math
import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import random
import scipy.integrate as integrate
from scipy.optimize import fsolve
from ..factor import Factor
import mpmath as mp
import cupy as cp
import json
from ..domain import Smoother
from bisect import bisect_left

mp.mp.dps = 1000

gpu = True
    

def read_csv(path, print_info=True):
    if print_info:
        print('    read csv data ' + path)
    data_list = []
    with open(path, 'r') as in_file:
        reader = csv.reader(in_file)
        headings = next(reader)
        data_list = [line for line in reader]
    return data_list, headings

def write_csv(data_list, headings, path, print_info=True):
    if print_info:
        print('    write csv data ' + path)
    with open(path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(headings)
        for line in data_list:
            writer.writerow(line)

# drop attributes in-place
def drop_attrs(data_list, headings, attr_list):
    for attr in attr_list:
        index = headings.index(attr)
        del headings[index]
        for line in data_list:
            del line[index]

def size(data):
    return reduce(lambda x,y: x*y, data, 1)

def dp_1norm(data, domain, index_list, marginal, noise, to_cpu=False):
    bins = domain.project(index_list).edge()
    histogram, _ = np.histogramdd(data[:, index_list], bins=bins)
    if gpu and not to_cpu:
        value = cp.asnumpy(marginal.values)
    else:
        value = marginal.values
    result1 = np.sum(np.abs(value - histogram))
    result2 =  (result1 + np.random.normal(scale=noise)) / 2 /len(data)
    return result1 / 2 /len(data) , result2

def print_data(np_data):
    for i in range(5):
        print_random_pos(np_data, 5)

def print_random_pos(np_data, print_num):
    start = random.randint(0, np_data.size-print_num)
    print(np_data.flatten()[start: start+print_num])

def dp_TVD(TVD_map, data, domain, index_list, noise):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in TVD_map:
        domain = domain.project(index_list)
        bins = domain.edge()
        size = domain.size()

        histogram, _= np.histogramdd(data[:, index_list], bins=bins)
        fact1 = Factor(domain, histogram, np)
        # print('!', domain.shape, '!', bins, '!', histogram.shape)

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
        TVD = np.sum(np.abs(fact4.values - fact1.values)) / 2 / data_num

        if gpu:
            TVD = TVD.item()
        noisy_TVD = TVD + np.random.normal(scale=noise)

        TVD_map[index_list] = [TVD, noisy_TVD]
        # print(index_list, TVD)

    return TVD_map[index_list]

def dp_mutual_info(MI_map, entropy_map, data, domain, index_list, noise):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in MI_map:
        temp_domain = domain.project(index_list)

        MI = -dp_entropy(entropy_map, data, domain, index_list, 0)[0]
        MI = +dp_entropy(entropy_map, data, domain, [index_list[0]], 0)[0]
        MI = +dp_entropy(entropy_map, data, domain, [index_list[1]], 0)[0]

        noisy_MI = MI + np.random.normal(scale=noise)

        MI_map[index_list] = [MI, noisy_MI]
        # print(index_list, MI)

    return MI_map[index_list]

def dp_entropy(entropy_map, data, domain, index_list, noise):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in entropy_map:
        temp_domain = domain.project(index_list)
        bins = temp_domain.edge()
        size = temp_domain.size()

        if len(index_list) <= 12 and size < 5e6:
            histogram, _= np.histogramdd(data[:, index_list], bins=bins)
            histogram = histogram.flatten()
            entropy = stats.entropy(histogram)
        else:
            value, counts = np.unique(data[:, index_list], return_counts=True, axis=0)
            entropy = stats.entropy(counts)

        noisy_entropy = entropy + np.random.normal(scale=noise)
        if entropy < 0:
            entropy = 0
        entropy_map[index_list] = [entropy, noisy_entropy]
        # if random.random() < 0.01:
        #     print(entropy)
    return entropy_map[index_list]

def dp_marginal_list(data, domain, marginal_list, noise, return_factor=True, valid=False, noise_type='normal'):
    marginal_dict = {}
    for marginal in marginal_list:
        temp_domain = domain.project(marginal)
        histogram, _ = np.histogramdd(data[:, marginal], bins=temp_domain.edge())
        if noise_type == 'normal':
            histogram += np.random.normal(scale=noise, size=temp_domain.shape)
        elif noise_type == 'Laplace':
            histogram += np.random.laplace(scale=noise, size=temp_domain.shape)
        else:
            print('invalid noise type')
            exit(-1)
        if valid:
            histogram[histogram<0] = 0
        if return_factor:
            fact = Factor(temp_domain, histogram)
            marginal_dict[tuple(marginal)] = fact
        else:
            marginal_dict[tuple(marginal)] = histogram
    return marginal_dict

def dp_smoothed_marginal(data, domain, marginal, noise, \
    return_factor=True, valid=False, smoother=None, noisy_smoother=-1):
    temp_domain = domain.project(marginal)
    histogram, _ = np.histogramdd(data[:, marginal], bins=temp_domain.edge())
    if smoother == None:
        if noisy_smoother < 0:
            # generate smoother using noisy free marginal distribution
            # if you have publicly known data, you can use this setting to generate and save a smoother
            # then use the smoother to smooth the private data
            # Caution: this histogram is not differentially private!
            smoother = Smoother(histogram, temp_domain, noise, 5*noise)
            marginal_noise = noise
        elif noisy_smoother < 1.0:
            # allocate noisy_smoother budget to generate a smoother
            smooth_noise = math.sqrt(1.0/noisy_smoother)*noise
            smoother = Smoother(histogram, temp_domain, smooth_noise, 5*smooth_noise)

            # use the rest of the budget to query marginal distribution
            marginal_noise = math.sqrt(1.0/(1-noisy_smoother))*noise
            
    histogram = smoother.smoothed_noisy_histogram(histogram, temp_domain, marginal_noise)

    if valid:
        histogram[histogram<0] = 0
    if return_factor:
        return Factor(temp_domain, histogram), smoother
    else:
        return histogram, smoother

# l1 sensitivity of entropy (for Laplace mechanism)
# ref: Plausible Deniability for Privacy-Preserving Data Synthesis
# def entropy_sensitivity(n):
#     return 1.0/n * (2.0 + 1/math.log(2.0) + 2.0*math.log2(n))

# l1 sensitivity of mutual information (for Laplace mechanism)
# ref: Differentially Private High-Dimensional Data Publication via Sampling-Based Inference
# def MI_sensitivity(n):
#     return 2.0/n * math.log2((n+1)/2) + (n-1)/n*math.log2((n+1)/(n-1))

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

# delta = 1e-10
    # epsilon = 0.1,    budget=0.00034033
    # epsilon = 1.0,    budget=0.02904375
    # epsilon = 10.0,   budget=2.14339730

# when x is larger than 5.0. quad exp(-t**2) requires very high accuracy
# since the integral is very close to 1 (maybe O(e^-x^3))
# that is, if x = 5, you will need mp.mp.dps > 5^3 = 125 to ensure the
# accuracy of add1 + exp(epsilon)*add2 ... of func
def erf_func(x):
    temp = 2.0/mp.sqrt(mp.pi)
    integral = mp.quad(lambda t: mp.exp(-t**2), [0, x])
    res = temp*integral
    return res

# you may need to tweak the start interval to find the root
# takes about 5 minutes
def cal_privacy_budget(epsilon, delta = 1e-5, error=1e-10):
    
    start = 0
    end = epsilon

    def func(x):
        if x <= 0:
            return - 2*delta
        add1 = erf_func(math.sqrt(x)/2/math.sqrt(2) - epsilon/math.sqrt(2*x))
        add2 = erf_func(math.sqrt(x)/2/math.sqrt(2) + epsilon/math.sqrt(2*x))
        res = add1 + mp.exp(epsilon)*add2 - mp.exp(epsilon) + 1 - 2*delta
        # print(add1, add2)
        return res
    # return mp.findroot(func, start, tol=1e-30)

    # gradient of func around its root is extemely small (maybe <= 1e-20 depending on epsilon)
    # which makes it is hard to set tol of mp.findroot and mp.mp.dps
    # we simply use binary search to ensure abs error of the root
    if func(start) > 0:
        start_geater = True
        if func(end) > 0:
            print('cant find root in given interval')
            exit(-1)
            return
    else:
        start_geater = False
        if func(end) < 0:
            print('cant find root in given interval')
            exit(-1)
            return
    
    while end - start > error:
        mid = (start + end)/2
        # print(mid)
        if func(mid) > 0:
            if start_geater:
                start = mid
            else:
                end = mid
        else:
            if start_geater:
                end = mid
            else:
                start = mid
    return (start + end)/2

# delta = 1e-5, use cal_privacy_budget to get these results
def privacy_budget(epsilon):
    f = {
        0.050: 0.0002996297553181648,
        0.060: 0.0004171067848801613,
        0.070: 0.0005520181730389595,
        0.080: 0.0007039634510874748,
        0.090: 0.0008725887164473534,
        0.095: 0.0009630667045712471,
        0.098: 0.0010192999616265297,
        0.100: 0.0010576052591204643,
        0.120: 0.0014757690951228142,
        0.140: 0.0019566575065255165,
        0.160: 0.002498799003660679,
        0.180: 0.003100908361375332,
        0.200: 0.003761877305805683,
        0.240: 0.005256538279354572,
        0.280: 0.006975927390158176,
        0.320: 0.008914261125028133,
        0.360: 0.011066538281738758,
        0.400: 0.013428307138383389,
        0.480: 0.01876466441899538,
        0.560: 0.02489518839865923,
        0.640: 0.031795711256563663,
        0.720: 0.039444915018975735,
        0.800: 0.047823661006987095,
        0.960: 0.0667015416547656,
        1.120: 0.08830544073134661,
        1.280: 0.11252749245613813,
        1.440: 0.13927177991718054,
        1.600: 0.16845174599438906,
        1.920: 0.23380873259156942,
        2.240: 0.3080356167629361,
        2.560: 0.39064376149326563,
        2.880: 0.48120140563696623,
        3.040: 0.5293390108272433,
        3.200: 0.5793221713975072
    }
    keys = list(f.keys())
    keys.sort()
    i = bisect_left(keys, epsilon)

    if abs(keys[i]-epsilon) < 1e-8:
        return f[keys[i]]
    if i-1 >= 0 and abs(keys[i-1]-epsilon) < 1e-8:
        return f[keys[i-1]]
    
    print('error: missing paramter of Gaussain mechanism', epsilon)
    exit(-1)

def contain_measure(measure_list, query_measure):
    for measure in measure_list:
        if len(measure) > 1 and isinstance(measure[1], float):
            print('error: call contain_measure using weighted measure')
            exit(-1)
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
    if gpu:
        prob = cp.asnumpy(prob)
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
    plt.clf()
    nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=20, font_size=4, width=0.5)
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['savefig.dpi'] = 600
    plt.show()
    plt.savefig(path)

def read_json_domain(domain_path):
    domain = json.load(open(domain_path, 'r'))
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

def exponential_mechanism(item_list, score_list, epsilon, sensitivity):
    debug_list = list(zip(item_list, score_list))

    score_list = np.array(score_list)
    score_list = score_list*epsilon/2/sensitivity
    # This is to avoid overflow. We do not actually need precision.
    score_list -= np.max(score_list)
    score_list = np.exp(score_list)
    score_list = score_list/np.sum(score_list)

    choice = np.random.choice(len(item_list), p=score_list)
    
    debug_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(len(debug_list)):
        item = debug_list[i]
        if item[0] == item_list[choice]:
            print('      Exponential mechanism select:')
            print('        select: {:d} total: {:d} score: {:.2e} max_score: {:.2e} budget/sensitivity: {:.2e}'\
                .format(i, len(debug_list), item[1], debug_list[0][1], epsilon/sensitivity))
            break

    return item_list[choice], i