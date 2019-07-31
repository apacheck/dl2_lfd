from abc import ABC
import numpy as np
import torch
from functools import reduce

EPS = 1e-8

def diffsat_theta(a, b, **kwargs):
    return torch.abs(a - b)

def diffsat_delta(a, b, **kwargs):
    return torch.sign(a - b) * diffsat_theta(a, b)

class Condition:

    def loss(self, **kwargs):
        return

    def satisfy(self, **kwargs):
        return

class BoolConst:

    def __init__(self, x):
        self.x = x.float()

    def loss(self):
        return 1.0 - self.x

    def satisfy(self):
        ret = self.x > (1 - EPS)
        return ret

class GT(Condition):
    """ a > b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def loss(self):
        return torch.clamp(diffsat_delta(self.b, self.a), min=0.0) + torch.eq(self.a, self.b).type(self.a.type())

    def satisfy(self):
        return self.a > self.b


class LT(Condition):
    """ a < b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def loss(self):
        return torch.clamp(diffsat_delta(self.a, self.b), min=0.0) + torch.eq(self.a, self.b).type(self.a.type())
            
    def satisfy(self):
        return self.a < self.b


class EQ(Condition):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def loss(self):
        return diffsat_theta(self.a, self.b)

    def satisfy(self):
        return torch.abs(self.a - self.b) < EPS

class GEQ(Condition):
    """ a >= b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def loss(self):
        return torch.clamp(diffsat_delta(self.b, self.a), min=0.0)

    def satisfy(self):
        return self.a >= self.b


class LEQ(Condition):
    """ a <= b """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def loss(self):
        return torch.clamp(diffsat_delta(self.a, self.b), min=0.0)

    def satisfy(self):
        return self.a <= self.b

class And(Condition):
    """ E_1 & E_2 & ... E_k """

    def __init__(self, exprs):
        self.exprs = exprs

    def loss(self):
        losses = [exp.loss() for exp in self.exprs]
        return reduce(lambda a, b: a + b, losses)

    def satisfy(self):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy()
            if not isinstance(sat, np.ndarray):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = ret * sat
        return ret

    
class Or(Condition):
    """ E_1 || E_2 || ... E_k """

    def __init__(self, exprs):
        self.exprs = exprs

    def loss(self):
        losses = [exp.loss() for exp in self.exprs]
        return reduce(lambda a, b: a * b, losses)

    def satisfy(self):
        ret = None
        for exp in self.exprs:
            sat = exp.satisfy()
            if not isinstance(sat, np.ndarray):
                sat = sat.cpu().numpy()
            if ret is None:
                ret = sat.copy()
            ret = np.maximum(ret, sat)
        return ret
 
class Implication(Condition):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.t = Or([Negate(a), b])

    def loss(self):
        return self.t.loss()

    def satisfy(self):
        return self.t.satisfy()

class Negate(Condition):

    def __init__(self, exp):
        self.exp = exp

        if isinstance(self.exp, LT):
            self.neg = GEQ(self.exp.a, self.exp.b)
        elif isinstance(self.exp, GT):
            self.neg = LEQ(self.exp, self.exp.b)
        elif isinstance(self.exp, EQ):
            self.neg = Or(LT(self.exp.a, self.exp.b), LT(self.exp.b, self.exp.a))
        elif isinstance(self.exp, LEQ):
            self.neg = GT(self.exp.a, self.exp.b)
        elif isinstance(self.exp, GEQ):
            self.neg = LT(self.exp.a, self.exp.b)
        elif isinstance(self.exp, And):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = Or(neg_exprs)
        elif isinstance(self.exp, Or):
            neg_exprs = [Negate(e) for e in self.exp.exprs]
            self.neg = And(neg_exprs)
        elif isinstance(self.exp, Implication):
            self.neg = And([self.exp.a, Negate(self.exp.b)])
        elif isinstance(self.exp, BoolConst):
            self.neg = BoolConst(1.0 - self.exp.x)
        else:
            assert False, 'Class not supported %s' % str(type(exp))

    def loss(self):
        return self.neg.loss()

    def satisfy(self):
        return self.neg.satisfy()







