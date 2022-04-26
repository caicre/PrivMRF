
gpu = False
# gpu = True

from .domain import Domain
import scipy
from collections import Iterable

import cupy as cp
import numpy as np

if gpu:
    xp = cp
else:
    xp = np

class Factor:
    def __init__(self, domain, values, xp=xp):
        self.domain = domain
        self.values = xp.array(values)
        self.size = len(self.domain)
    
    def to_cpu(self):
        self.values = cp.asnumpy(self.values)
        return self

    def __add__(self, parameter):
        if not isinstance(parameter, Factor):
            return Factor(self.domain, self.values + parameter, cp.get_array_module(self.values))
        domain = self.domain + parameter.domain
        add1 = self.expand(domain)
        add2 = parameter.expand(domain)
        return Factor(domain, add1.values + add2.values, cp.get_array_module(self.values))

    def __mul__(self, parameter):
        if not isinstance(parameter, Factor):
            return Factor(self.domain, self.values * parameter, cp.get_array_module(self.values))
        domain = self.domain + parameter.domain
        mul1 = self.expand(domain)
        mul2 = parameter.expand(domain)
        return Factor(domain, mul1.values * mul2.values, cp.get_array_module(self.values))

    def __rmul__(self, parameter):
        return self.__mul__(parameter)

    def __imul__(self, parameter):
        self.values *= parameter
        return self

    def __truediv__(self, parameter):
        if isinstance(parameter, Factor):
            parameter = Factor(parameter.domain, 1/parameter.values, cp.get_array_module(self.values))
            return self * parameter
        return self * (1/parameter)
    
    def __sub__(self, parameter):
        if isinstance(parameter, Factor):
            domain = self.domain + parameter.domain
            sub1 = self.expand(domain)
            sub2 = parameter.expand(domain)
            
            return Factor(domain, sub1.values - sub2.values, cp.get_array_module(self.values))
        else:
            return Factor(self.domain, self.values - parameter, cp.get_array_module(self.values))

    def sum(self):
        return cp.get_array_module(self.values).sum(self.values)

    @staticmethod
    def zeros(domain, xp=xp):
        return Factor(domain, xp.zeros(domain.shape), xp)

    def expand(self, domain):
        if len(domain) == len(self.domain):
            return self
        assert(set(self.domain.dict.keys()) <= set(domain.dict.keys()))
        shape = self.domain.shape + [1] * (len(domain) - len(self.domain))
        
        index_list = domain.index_list(self.domain)

        # print(domain.attr_list, domain.shape)
        # print(shape)
        # print(self.domain.attr_list, self.domain.shape, self.values.shape)

        values = self.values.reshape(shape)
        values = cp.get_array_module(self.values).moveaxis(values, range(len(self.domain)), index_list)
        values = cp.get_array_module(self.values).broadcast_to(values, domain.shape)

        return Factor(domain, values, cp.get_array_module(self.values))

    def logsumexp(self, attr_set=None):
        if attr_set == None or len(attr_set) == 0:
            xp = cp.get_array_module(self.values)
            if xp == cp:
                values = xp.exp(self.values)
                values = xp.sum(values)
                values = xp.log(values)
                return values
            else:
                return scipy.special.logsumexp(self.values)
        assert(set(attr_set) <= set(self.domain.attr_list))
        sum_attr = list(set(self.domain.attr_list) - set(attr_set))
        sum_attr = tuple(self.domain.index_list(sum_attr))
        if cp.get_array_module(self.values) == cp:
            values = cp.exp(self.values)
            values = cp.sum(values, axis=sum_attr)
            values = cp.log(values)
        else:
            values = scipy.special.logsumexp(self.values, axis=sum_attr)
        return Factor(self.domain.project(attr_set), values, cp.get_array_module(self.values))

    def exp(self):
        return Factor(self.domain, cp.get_array_module(self.values).exp(self.values), cp.get_array_module(self.values))

    def project(self, domain):
        if not isinstance(domain, Domain):
            if not isinstance(domain, Iterable):
                domain = [domain]
            domain = self.domain.project(domain)
        assert(set(domain.attr_list) <= set(self.domain.attr_list))
        new_domain = self.domain.invert(domain)
        index_list = tuple(self.domain.index_list(new_domain))

        values = cp.get_array_module(self.values).sum(self.values, axis=index_list)
        return Factor(domain, values, cp.get_array_module(self.values))

    def copy(self):
        return Factor(self.domain, self.values.copy(), cp.get_array_module(self.values))

    def moveaxis(self, attr_list):
        new_domain = self.domain.moveaxis(attr_list)
        index_list = tuple(new_domain.index_list(self.domain))
        values = cp.get_array_module(self.values).moveaxis(self.values, range(len(self.domain)), index_list)
        return Factor(new_domain, values, cp.get_array_module(self.values))

class Potential(dict):
    def __init__(self, factor_dict):
        self.factor_dict = factor_dict
        dict.__init__(self, factor_dict)
        
    def __sub__(self, potential):
        assert(len(self) == len(potential))
        ans = {clique: self[clique] - potential[clique] for clique in self}
        return Potential(ans)

    def __add__(self, potential):
        assert(len(self) == len(potential))
        ans = {clique: self[clique] + potential[clique] for clique in self}
        return Potential(ans)

    def __mul__(self, parameter):
        return Potential({clique: parameter*self[clique] for clique in self})

    def __rmul__(self, parameter):
        return self.__mul__(parameter)

    def __imul__(self, parameter):
        for clique in self:
            self[clique] *= parameter
        return self

    def dot(self, potential):
        for clique in self:
            xp = cp.get_array_module(self[clique].values)
            break
        return sum(xp.sum((self[clique] * potential[clique]).values) for clique in self)

    def copy(self):
        return Potential({clique: self[clique].copy() for clique in self})

    @staticmethod
    def l2_marginal_loss(marginal_potential, measure_potential):
        gradient = marginal_potential - measure_potential
        loss = 1/2 * gradient.dot(gradient)
        return loss.item(), gradient