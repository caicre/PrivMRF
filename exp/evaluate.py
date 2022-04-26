import csv
import PrivMRF.utils as utils
import numpy as np
from PrivMRF.domain import Domain
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
import json
import pandas as pd
from PrivMRF.main import run_syn
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PrivMRF.preprocess import read_preprocessed_data
from sklearn import preprocessing
import os
import multiprocessing as mp


# shuffle and split data
def split(data_name):
    print('  split data')
    data, headings = utils.tools.read_csv('./preprocess/'+data_name+'.csv')
    data = np.array(data, dtype=int)

    np.random.shuffle(data)

    path = './exp_data'
    if not os.path.exists(path):
        os.mkdir(path)

    utils.tools.write_csv(data[:int(0.8*len(data))], headings, './exp_data/'+data_name+'_train.csv')
    utils.tools.write_csv(data[int(0.8*len(data)):], headings, './exp_data/'+data_name+'_test.csv')

# evaluate dp data on k way marginal task
def k_way_marginal(data_name, dp_data_list, k, marginal_num):
    # data, headings = utils.tools.read_csv('./exp_data/' + data_name + '_train.csv')
    data, headings = utils.tools.read_csv('./preprocess/' + data_name + '.csv', print_info=False)
    data = np.array(data, dtype=int)

    attr_num = data.shape[1]
    data_num = data.shape[0]
    # attr_num = 10

    domain = json.load(open('./preprocess/'+data_name+'.json'))
    domain = {int(key): domain[key] for key in domain}
    domain = Domain(domain, list(range(attr_num)))

    marginal_list = [tuple(sorted(list(np.random.choice(attr_num, k, replace=False)))) for i in range(marginal_num)]
    marginal_dict = {}
    size_limit = 1e8
    for marginal in marginal_list:
        temp_domain = domain.project(marginal)
        if temp_domain.size() < size_limit:
            # It is fast when domain is small, howerver it will allocate very large array
            edge = temp_domain.edge()
            histogram, _ = np.histogramdd(data[:, marginal], bins=edge)
            marginal_dict[marginal] = histogram
        else:
            uniques, cnts = np.unique(data, return_counts=True, axis=0)
            uniques = [tuple(item) for item in uniques]
            cnts = [int(item) for item in cnts]
            marginal_dict[marginal] =  dict(zip(uniques, cnts))

    # total variation distance
    tvd_list = []
    # for dp_data_path in dp_data_list:
    for dp_data in dp_data_list:
        # dp_data, headings = utils.tools.read_csv(dp_data_path, print_info=False)
        dp_data = np.array(dp_data, dtype=int)
        dp_data_num = len(dp_data)
        tvd = 0
        # print('data:', dp_data_path)
        for marginal in marginal_dict:
            temp_domain = domain.project(marginal)
            if temp_domain.size() < size_limit:
                edge = temp_domain.edge()
                histogram, _ = np.histogramdd(dp_data[:, marginal], bins=edge)
                histogram *= data_num/dp_data_num
                temp_tvd = np.sum(np.abs(marginal_dict[marginal] - histogram)) / len(data) / 2
            else:
                uniques, cnts = np.unique(dp_data, return_counts=True, axis=0)
                uniques = [tuple(item) for item in uniques]
                cnts = [int(item)*data_num/dp_data_num for item in cnts]
                diff = []
                unique_cnt = marginal_dict[marginal]
                for i in range(len(uniques)):
                    if uniques[i] in unique_cnt:
                        diff.append(cnts[i] - unique_cnt[uniques[i]])
                    else:
                        diff.append(cnts[i])
                diff = np.array(diff)
                # TVD = 1/2 * sum(abs(diff)) = 1.0 * sum(max(diff, 0))
                diff[diff<0] = 0
                temp_tvd = np.sum(diff)/len(data)

            if temp_tvd > 1:
                print(marginal, temp_domain.size(), temp_tvd)
            tvd += temp_tvd
            # print('    {}: {}'.format(marginal, temp_tvd))
        tvd /= len(marginal_dict)
        tvd_list.append(tvd)
    return tvd_list

# evaluate dp data on SVM classifier task
def svm_classifier(exp_name, data_name, dp_data_list, classifier_num, target_list, \
    print_lock, result_lock, epsilon, cross_valid_round,\
        target_parallel):

    data = []
    for k in range(5):
        if k == cross_valid_round:
            continue
        data_list, _ = utils.tools.read_csv('./exp_data/'+data_name+str(k)+'.csv')
        data.append(data_list)
    data = [np.array(temp, dtype=int) for temp in data]
    data = np.concatenate(data, axis=0)

    data =  pd.DataFrame(data)
    attr_num = data.shape[1]

    test_data, headings = utils.tools.read_csv('./exp_data/'+data_name+str(cross_valid_round)+'.csv')
    test_data = np.array(test_data, dtype=int)
    test_data = pd.DataFrame(test_data)

    data_list = []
    for j in range(len(dp_data_list) + 1):
        if j == 0:
            # data_list.append(data)
            pass
        else:
            dp_data, headings = utils.tools.read_csv(dp_data_list[j-1])
            dp_data = np.array(dp_data, dtype=int)

            dp_data = pd.DataFrame(dp_data)

            data_list.append(dp_data)

    mis_rate = [0] * len(data_list)
    classifier_num = min(classifier_num, attr_num, len(target_list))

    # for i in range(1):
    for i in range(classifier_num):
        target = target_list[i]
        for j in range(len(data_list)):
            train_data = data_list[j]

            X = train_data.loc[:, train_data.columns != target]
            Y = train_data.loc[:, target]

            # fit
            # clf = svm.SVC(gamma='auto', verbose=False, max_iter=100000)
            clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
            # clf = svm.LinearSVC(verbose=True, max_iter=10000, dual=False)
            # clf = svm.LinearSVC()
            start_time = time.time()
            clf.fit(X, Y)

            # test
            test_X = test_data.loc[:, test_data.columns != target]
            test_Y = test_data.loc[:, test_data.columns == target]
                        
            test_predict = clf.predict(test_X)
            acc = accuracy_score(test_Y, test_predict)

            mis_rate[j] += 1-acc
            print_lock.acquire()
            print('\t\tclassifier: {}/{} target: {} mis rate: {:.4f} time: {:.4f}'\
                .format(i, classifier_num, target, 1-acc, time.time()-start_time))
            print_lock.release()
    mis_rate = [mis/classifier_num for mis in mis_rate]

    # data_name = 'adult1'
    result_lock.acquire()
    if os.path.exists('./result/'+exp_name+'_SVM.json'):
        with open('./result/'+exp_name+'_SVM.json', 'r') as in_file:
            result = json.load(in_file)
    else:
        result = {}
    if str(epsilon) not in result:
        result[str(epsilon)] = {}
    if data_name not in result[str(epsilon)]:
        result[str(epsilon)][data_name] = {}

    if target_parallel:
        if str(cross_valid_round) not in result[str(epsilon)][data_name]:
            result[str(epsilon)][data_name][str(cross_valid_round)] = {}
        result[str(epsilon)][data_name][str(cross_valid_round)][target_list[0]] = mis_rate
    else:
        result[str(epsilon)][data_name][str(cross_valid_round)] = mis_rate

    with open('./result/'+exp_name+'_SVM.json', 'w') as out_file:
        json.dump(result, out_file)
    result_lock.release()

    
def merge_json(dict1, dict2, update=False):
    def merge_dict(a, b):
        if not isinstance(a, dict):
            # print('result conflict', a, b)
            if update:
                return b
        for item in b:
            if item in a:
                a[item] = merge_dict(a[item], b[item])
            else:
                a[item] = b[item]
        return a
    return merge_dict(dict1, dict2)

# main function for running experiments
def run_experiment(data_list, method_list, exp_name, \
    epsilon_list, task='TVD', repeat=10, marginal_num=300, \
        classifier_num=10, generate=True):
    if task == 'TVD':
        result = marginal_exp(data_list, method_list, exp_name, \
            epsilon_list, repeat=repeat, marginal_num=marginal_num, \
                classifier_num=classifier_num)
        path = './result/'+exp_name+'_TVD.json'

        print(result)
        json.dump(result, open(path, 'w'))

    else:
        svm_exp(data_list, method_list, exp_name, \
            epsilon_list, repeat=repeat, marginal_num=marginal_num, \
                classifier_num=classifier_num, generate=generate)
        # the result is in './result/'+exp_name+'_SVM.json'

def cross_validation_data(data_name):
    full_data, _, _ = read_preprocessed_data(data_name)
    full_data = np.array(full_data, dtype=int)
    np.random.shuffle(full_data)
    headings = list(range(full_data.shape[1]))

    for j in range(5):
        start = int(j/5 * len(full_data))
        end = int((j+1)/5 * len(full_data))

        utils.tools.write_csv(full_data[start: end], headings, './exp_data/'+data_name+str(j)+'.csv')
                
        
def svm_exp(data_name_list, method_list, exp_name, \
    epsilon_list, repeat=10, marginal_num=300, classifier_num=10, generate=True):
    result = {}
    # generate target attribute list, which should be performed only once
    target_dict = {}
    for data_name in data_name_list:
        domain_json = json.load(open('./preprocess/'+data_name+'.json'))
        target_list = list(range(len(domain_json)))
        random.shuffle(target_list)
        target_dict[data_name] = target_list

        if generate:
            cross_validation_data(data_name)

    json.dump(target_dict, open('./exp_data/target.json', 'w'))
    target_dict = json.load(open('./exp_data/target.json'))

    if generate:
        for epsilon in epsilon_list:
            print('epsilon {}'.format(epsilon))
            result[str(epsilon)] = {}
            for data_name in data_name_list:
                print('  data {}'.format(data_name))
                result[str(epsilon)][data_name] = {}
                for i in range(repeat):
                    result[str(epsilon)][data_name][str(i)] = {}
                    print('    repeat {}/{}'.format(i, repeat))
                    print('    method {}'.format(method_list))

                    full_data = []
                    for k in range(5):
                        data_list, _ = utils.tools.read_csv('./exp_data/'+data_name+str(k)+'.csv')
                        full_data.append(data_list)
                    full_data = [np.array(temp, dtype=int) for temp in full_data]

                    for j in range(0, 5):
                    # for j in range(4, 5):
                        print('cross validation: {}/{}'.format(j, 5))
                        headings = list(range(len(full_data[0][0])))

                        train_data = [full_data[temp] for temp in range(5) if temp != j]
                        train_data = np.concatenate(full_data, axis=0)
                        pd.DataFrame(train_data).to_csv('./exp_data/'+data_name+'_train.csv', header=headings, index=None)

                        dp_data_list = []
                        for method in method_list:
                            temp_exp_name = exp_name+str(epsilon)+'_'+str(j)

                            if method == 'PrivMRF':
                                dp_data_list.append('./out/'+method+'_'+data_name+'_'+temp_exp_name+'.csv')
                                run_syn(data_name, temp_exp_name, epsilon, task='SVM')
    else:
        print_lock = mp.Lock()
        result_lock = mp.Lock()

        # train all attrubute in parallel when this is true
        # note it will train every attribute, dataset, epsilon, cross validation in parallel
        # open it in case you have attribute num x dataset x epsilon x cross validation cores
        target_parallel = False
                        
        for epsilon in epsilon_list:
            print_lock.acquire()
            print('epsilon {}'.format(epsilon))
            print_lock.release()

            for data_name in data_name_list:
                print_lock.acquire()
                print('  data {}'.format(data_name))
                print_lock.release()

                full_data = []
                for k in range(5):
                    data_list, _ = utils.tools.read_csv('./exp_data/'+data_name+str(k)+'.csv')
                    full_data.append(data_list)
                full_data = [np.array(temp, dtype=int) for temp in full_data]

                for j in range(0, 5):
                    dp_data_list = []
                    for method in method_list:
                        
                        temp_exp_name = exp_name+str(epsilon)+'_'+str(j)
                            
                        dp_data_list.append('./out/'+method+'_'+data_name+'_'+temp_exp_name+'.csv')

                    if target_parallel:
                        for i in range(len(target_dict[data_name])):

                            proc = mp.Process(target=svm_classifier, \
                                args=(exp_name, data_name, dp_data_list, classifier_num, [target_dict[data_name][i]],\
                                    print_lock, result_lock, epsilon, j, target_parallel))
                            proc.start()
                    else:
                        proc = mp.Process(target=svm_classifier, \
                            args=(exp_name, data_name, dp_data_list, classifier_num, target_dict[data_name],\
                                print_lock, result_lock, epsilon, j, target_parallel))
                        proc.start()

        
def marginal_exp(data_list, method_list, exp_name, \
    epsilon_list, repeat=10, marginal_num=300, classifier_num=10):

    # ways = [3,]
    ways = [3, 4, 5]
    for epsilon in epsilon_list:
        
        for data_name in data_list:
            
            # for i in range(8, 10):
            for i in range(repeat):
                
                print('epsilon {}'.format(epsilon))
                print('  data {}'.format(data_name))
                print('    repeat {}/{}'.format(i, repeat))
                print('    method {}'.format(method_list))

                dp_data_list = []
                for method in method_list:
                    temp_exp_name = exp_name
                    if method == 'PrivMRF':
                        dp_data_list.append(run_syn(data_name, temp_exp_name, epsilon, task='TVD'))

                for k in ways:
                    tvd_list = k_way_marginal(data_name, dp_data_list, k, marginal_num)
                    print('      {} way marginal {}'.format(k, tvd_list))

                    if os.path.exists('./result/'+temp_exp_name+'_log.json'):
                        with open('./result/'+temp_exp_name+'_log.json', 'r') as in_file:
                            result = json.load(in_file)
                    else:
                        result = {}
                    if str(epsilon) not in result:
                        result[str(epsilon)] = {}
                    if data_name not in result[str(epsilon)]:
                        result[str(epsilon)][data_name] = {}
                    if str(i) not in result[str(epsilon)][data_name]:
                        result[str(epsilon)][data_name][str(i)] = {}
                    result[str(epsilon)][data_name][str(i)][str(k)] = tvd_list

                    with open('./result/'+temp_exp_name+'_log.json', 'w') as out_file:
                        json.dump(result, out_file)

    average = {}
    for epsilon in result:
        average[epsilon] = {}
        for data_name in result[epsilon]:
            average[epsilon][data_name] = {}
            for k in ways:
                temp = []
                for i in result[epsilon][data_name]:
                    temp.append(result[epsilon][data_name][i][str(k)])
                temp = np.array(temp)
                temp = np.sum(temp, axis=0)/len(temp)
                average[epsilon][data_name][str(k)] = list(temp)

    print(average)
    return average
