#! coding:utf-8
# tools.py


import unittest
from operator import attrgetter
from copy import deepcopy
import random

import numpy as np


class Individuale(object):
    def __init__(self, pop):
        self._x = np.asarray(pop)
        self._fitness_values = None
        self._fitness_valid = False

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = np.asarray(v)

    @x.deleter
    def x(self):
        self._x = None

    @property
    def fitness_values(self):
        return self._fitness_values

    @fitness_values.setter
    def fitness_values(self, v):
        self._fitness_values = v
        self._fitness_valid = True

    @fitness_values.deleter
    def fitness_values(self):
        self._fitness_values = None
        self._fitness_valid = False

    @property
    def fitness_valid(self):
        return self._fitness_valid

    @fitness_valid.setter
    def fitness_valid(self, v):
        v = bool(v)
        self._fitness_valid = v

    @fitness_valid.deleter
    def fitness_valid(self):
        self._fitness_valid = False

    def __str__(self):
        return "{} {} {}".format(self.x, self.fitness_values, self.fitness_valid)

    def __repr__(self):
        return "x:{}, fittness.values:{} fitness_valid{}".format(self.x, self.fitness_values, self.fitness_valid)


def converter(min=None, max=None, n=None, type=None):
    """実パラメータ空間の値を，指定された整数範囲に射影する
    :min, max: パラメータ空間の最小値，最大値
    :n: 離散化階数
    :type: 射影先の型. int:[0, n-1], float:[0.0, 1.0], bool:[0,1]
    """
    if type == bool:
        return int, bool, [0, 1]

    else:
        x1, x2 = min, max

        if type == int:
            y1, y2 = 0, n - 1
            init = range(n)
        elif type == float:
            y1, y2 = 0.0, 1.0
            init = np.linspace(0, 1, n).tolist()

        a = float(y1 - y2) / float(x1 - x2)
        b = y1 - float(y1 - y2) / float(x1 - x2) * x1

        def encoder(x):
            if not min <= x <= max:
                raise Warning("Input Value is Cliped => {:.2f}<={:.2f}<={:.2f}".format(x1, x, x2))
                pass
            return a * x + b

        def decoder(y):
            if not y1 <= y <= y2:
                raise Warning("Input Value is Cliped => {:.2f}<={:.2f}<={:.2f}".format(y1, y, y2))
                pass
            return (y - b) / a

        return encoder, decoder, init


def decoder(decs):
    """遺伝子の実数型から，パラメータ空間の型へ戻すためのデコーダー
    :decs: <list> tools.convrterから取得したdecoderのリスト
    """
    assert isinstance(decs, list)

    def closure(ys):
        """
        格遺伝子の値をデコードする
        :ys: <list> 遺伝子の実数値.
        """
        ret = []
        for dec, y in zip(decs, ys):
            ret.append(dec(y))
        return ret

    return closure


def encoder(encd):
    """未実装"""
    assert isinstance(encd, list)

    def closure(xlist):
        """
        格遺伝子の値をデコードする
        :xlist: <list> 遺伝子の実数値.
        """
        ret = []
        for enc, x in zip(encd, xlist):
            ret.append(enc(x))
        return ret

    return closure


def population(params, n=100):
    """
    p1 = tools.paramgen(
    name="HightPower.f1", min=1000, max=20000, delta=1000, type=int)
    """
    pops = []
    for i in range(n):
        pop = []
        for param in params:
            pop.append(np.random.choice(param))

        individuale = Individuale(pop)
        pops.append(individuale)
        # print("No.{} : {} : {}".format(i, individuale, individuale.x))
    return pops


def param_generater(**p):
    """initial population value generater"""
    if p["type"] == int:
        return np.arange(p["min"], p["max"] + p["delta"], p["delta"], np.int)
    elif p["type"] == float:
        return np.arange(p["min"], p["max"] + p["delta"], p["delta"], np.float)
    elif p["type"] == bool:
        return np.array([0, 1])
    else:
        raise TypeError


def evalute(pop):
    # return pop.x.sum()
    return sum(pop.x)


def select(individuals, n):
    """Individuales Selector
    same selBest()
    """
    # return selBest(individuals, n)
    return individuals[:n]


def selBest(individuals, n):
    """Individuales Selector"""
    return sorted(individuals, key=attrgetter("fitness_values"), reverse=True)[:n]


def selTournamentDCD(individuals, k):
    if k % 4 != 0:
        raise ValueError

    def tourn1d(ind1, ind2):
        if ind1.fitness_values > ind1.fitness_values:
            return ind1
        elif ind1.fitness_values < ind1.fitness_values:
            return ind2
        # same value
        if random.random() <= 0.5:
            return ind1
        else:
            return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in xrange(0, k, 4):
        chosen.append(tourn1d(individuals_1[i], individuals_1[i + 1]))
        chosen.append(tourn1d(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(tourn1d(individuals_2[i], individuals_2[i + 1]))
        chosen.append(tourn1d(individuals_2[i + 2], individuals_2[i + 3]))

    chosen = map(clone, chosen)

    return chosen


def mate(ind1, ind2):
    raise StandardError


def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1.x), len(ind2.x))
    cxpoint = random.randint(1, size - 1)
    ind1.x[cxpoint:], ind2.x[cxpoint:] = ind2.x[cxpoint:], ind1.x[cxpoint:]

    return ind1, ind2


def mutate(ind1):
    """mutation base method"""
    raise StandardError


def mutUniformInt(individual, low, up, idps):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual.x)

    for i, xl, xu in zip(xrange(size), low, up):
        individual.x[i] = random.randint(xl, xu)

    return individual,


def clone(individuals):
    """object copy"""
    return deepcopy(individuals)


class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def test_Individuales(self):
        pop = [1, 2, 3, 4, 5]
        ind = Individuale(pop)
        self.assertEqual(ind.x.all(), np.asarray(pop).all())
        self.assertIsNone(ind.fitness_values)
        self.assertFalse(ind.fitness_valid)

        ind.fitness_values, ind.fitness_valid = 999, True
        self.assertIsNotNone(ind.fitness_values)
        self.assertTrue(ind.fitness_valid)

        del ind.fitness_values, ind.fitness_valid
        self.assertIsNone(ind.fitness_values)
        self.assertFalse(ind.fitness_valid)

    def test_population(self):
        p1_enc, p1_dec, p1_init = converter(min=100, max=20000, n=10, type=int)  # 0-9
        self.assertEqual(p1_enc(100), 0)
        self.assertAlmostEqual(p1_enc(20000), 10 - 1)
        self.assertEqual(len(p1_init), 10)

        p2_enc, p2_dec, p2_init = converter(min=0.1, max=1, n=10, type=int)  # 0-9
        self.assertEqual(p2_enc(0.1), 0)
        self.assertAlmostEqual(p2_enc(1), 10 - 1)
        self.assertEqual(len(p2_init), 10)

        p3_enc, p3_dec, p3_init = converter(min=10000, max=200000, n=10, type=float)  # 0-9
        self.assertEqual(p3_enc(10000), 0)
        self.assertAlmostEqual(p3_enc(200000), 1)
        self.assertEqual(len(p3_init), 10)

        p4_enc, p4_dec, p4_init = converter(type=bool)  # 0-9
        self.assertEqual(p4_enc(True), 1)
        self.assertEqual(p4_enc(False), 0)
        self.assertEqual(p4_enc(1), True)
        self.assertEqual(p4_enc(0), False)
        self.assertEqual(len(p4_init), 2)

        # p1_enc(100) # 0
        # p1_enc(1000) # 9
        # p1_dec(0) # 100
        # p1_dec(1) # 1000
        # map([p1_enc, p2_enc, p3_enc, p4_enc], ind.x)

        NPOP = 100
        # population init
        pops = population([p1_init, p2_init, p3_init, p4_init], n=NPOP)
        self.assertIsInstance(pops, list)
        self.assertEqual(len(pops), NPOP)
        self.assertIsInstance(pops[0], Individuale)


class TestTools2(unittest.TestCase):
    def setUp(self):
        p1_enc, p1_dec, p1_init = converter(min=100, max=20000, n=10, type=int)  # 0-9
        p2_enc, p2_dec, p2_init = converter(min=0.1, max=1, n=10, type=int)  # 0-9
        p3_enc, p3_dec, p3_init = converter(min=10000, max=200000, n=10, type=int)  # 0-9
        p4_enc, p4_dec, p4_init = converter(type=bool)  # 0-9
        NPOP = 100
        self.pops = population([p1_init, p2_init, p3_init, p4_init], n=NPOP)

        self.low = [0, 0, 0, 0]
        self.up = [9, 9, 9, 1]

    def test_select(self):
        pops = self.pops
        # selection : selBest()
        selected_pops = selBest(pops, 11)
        self.assertEqual(len(selected_pops), 11)
        self.assertIsInstance(selected_pops[0], Individuale)
        self.assertTrue(selected_pops[0].fitness_values >= selected_pops[1].fitness_values)
        self.assertTrue(selected_pops[0].fitness_values >= selected_pops[3].fitness_values)
        self.assertTrue(selected_pops[0].fitness_values >= selected_pops[5].fitness_values)
        # selection : select()
        selected_pops = select(pops, 11)
        self.assertEqual(len(selected_pops), 11)
        self.assertIsInstance(selected_pops[0], Individuale)

    def test_selTournamentDCD(self):
        pops = self.pops
        # selection : selTournamentDCD()
        selected_pops = selTournamentDCD(pops, 4 * 25)
        self.assertEqual(len(selected_pops), 4 * 25)
        self.assertIsInstance(selected_pops[0], Individuale)

        with self.assertRaises(ValueError):
            selTournamentDCD(pops, 5)
            selTournamentDCD(pops, 7)
            selTournamentDCD(pops, 15)
            selTournamentDCD(pops, 17)

    def test_evalution(self):
        pops = self.pops
        # evalution : Reset
        for p in pops:
            del p.fitness_values
            del p.fitness_valid
        # select
        selected_pops = selTournamentDCD(pops, 16)

        # evalution
        for p in pops:
            p.fitness_values = evalute(p)
            p.fitness_valid = True
        # select
        selected_pops = selTournamentDCD(pops, 16)
        self.assertEqual(len(selected_pops), 16)
        self.assertIsInstance(selected_pops[0], Individuale)

    def test_clone(self):
        pops = self.pops
        # copy
        offsprint = selTournamentDCD(pops, len(pops))
        offsprint_cp = clone(offsprint)
        # offsprint_cp = list(map(clone, offsprint))
        for offsp, offsp_cp in zip(offsprint, offsprint_cp):
            self.assertEqual(offsp.fitness_values, offsp_cp.fitness_values)
            self.assertEqual(offsp.fitness_valid, offsp_cp.fitness_valid)
            self.assertNotEqual(id(offsp), id(offsp_cp))
            self.assertNotEqual(offsp, offsp_cp)

            offsp_cp.fitness_values = 10
            offsp.fitness_values = 1000000000
            self.assertNotEqual(offsp.fitness_values, offsp_cp.fitness_values)
            self.assertEqual(offsp.fitness_valid, offsp_cp.fitness_valid)
            self.assertNotEqual(id(offsp), id(offsp_cp))
            self.assertNotEqual(offsp, offsp_cp)

    def test_mate(self):
        pops = self.pops

        mate = cxOnePoint
        offspring = selTournamentDCD(pops, len(pops))
        offspring = list(map(clone, offspring))

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            mate(ind1, ind2)
            self.assertIsInstance(ind1, Individuale)
            self.assertIsInstance(ind2, Individuale)
            del ind1.fitness_values, ind1.fitness_valid
            del ind2.fitness_values, ind2.fitness_valid

        for ind in offspring:
            self.assertIsNone(ind.fitness_values)
            self.assertFalse(ind.fitness_valid)

    def test_mutate(self):
        pops = self.pops

        low = self.low
        up = self.up
        indpb = 0.02

        offspring = selTournamentDCD(pops, len(pops))
        offspring = list(map(clone, offspring))

        print("-- mutation start --")
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            mutUniformInt(ind1, low, up, indpb)
            mutUniformInt(ind2, low, up, indpb)

            del ind1.fitness_values, ind1.fitness_valid
            del ind2.fitness_values, ind2.fitness_valid

            for v, l, u in zip(ind1.x, low, up):
                self.assertTrue(l <= v <= u)

            for v, l, u in zip(ind2.x, low, up):
                self.assertTrue(l <= v <= u)

        for ind in offspring:
            self.assertIsNone(ind.fitness_values)
            self.assertFalse(ind.fitness_valid)


class TestTools3(unittest.TestCase):
    def setUp(self):
        # Init Individual Value
        PMAX = 32
        p1 = converter(min=1000, max=10000, n=PMAX, type=int)  # 0-9
        p2 = converter(min=10, max=20, n=PMAX, type=float)  # 0-9
        p3 = converter(type=bool)  # 0-9
        convs = [p1, p2, p3]
        init_params, decs, encs = list(), list(), list()
        for enc, dec, init in convs:
            encs.append(enc)
            decs.append(dec)
            init_params.append(init)

        self.encs = encs
        self.decs = decs

    def test_encoder(self):
        """パラメータ値を離散値へエンコード"""
        _encoder = encoder(self.encs)

        self.assertAlmostEqual(_encoder([1000, 20, 0]), [0, 1.0, False])
        self.assertAlmostEqual(_encoder([10000, 20, 0]), [32 - 1, 1.0, False])
        self.assertAlmostEqual(_encoder([1000, 10, 0]), [0, 0.0, False])
        self.assertAlmostEqual(_encoder([1000, 20, 0]), [0, 1.0, False])
        self.assertAlmostEqual(_encoder([1000, 20, 1]), [0, 1.0, True])

        with self.assertRaises(Warning):
            y1 = [999, 10, 9]
            ans1 = _encoder(y1)

    def test_decoder(self):
        """離散値をパラメータ値へデコード"""
        _decoder = decoder(self.decs)

        self.assertAlmostEqual(_decoder([32 - 1, 1.0, 1]), [10000, 20, True])
        # self.assertAlmostEqual(_decoder([0, 1.0, 1]), [1000., 20., True])

        with self.assertRaises(Warning):
            self.assertEqual(_decoder([32, 2, 1]), [10000, 20, True])


if __name__ == '__main__':
    unittest.main()
