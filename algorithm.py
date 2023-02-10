
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score, roc_auc_score
from sklearn.cluster import MeanShift, KMeans, OPTICS

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
# from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination, get_selection
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import MaximumGenerationTermination
from pymoo.util.ref_dirs import get_reference_directions
from operator import itemgetter
from collections import Counter

import visualisation
import pandas as pd
import pickle
import os


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def taxicab_sample(n, r):
    sample = []

    for _ in range(n):
        spread = (r**2 - np.sum([x ** 2 for x in sample])) ** (1/2)
        sample.append(spread * (2 * np.random.rand() - 1))

    return np.random.permutation(sample)


class HybridCenterSampling:
    def __init__(self, rads=0.25, fracs=1, centers=None, scaling=0.0, n=None):
        if not isinstance(rads, np.ndarray):
            self.rads = np.array([rads for i in range(self._minority.shape[0])])
        else:
            self.rads = rads

        if not isinstance(fracs, np.ndarray):
            self.fracs = np.array([fracs for i in range(self._minority.shape[0])])
        else:
            self.fracs = fracs
        self._centers = centers
        self.scaling = scaling
        self.n = n
        self.appended = None

    def set_distances(self):

        self._classes = np.unique(self._y)
        sizes = [sum(self._y == c) for c in self._classes]

        assert len(self._classes) == len(set(sizes)) == 2

        self.minority_class = self._classes[np.argmin(sizes)]
        self.majority_class = self._classes[np.argmax(sizes)]
        self._minority = self._X[self._y == self.minority_class]
        self._majority = self._X[self._y == self.majority_class]

        if self.n is None:
            self._n = len(self._majority) - len(self._minority)

        if self._centers is not None:
            self._distances = np.zeros((len(self._centers), len(self._majority)))

            for i in range(len(self._centers)):
                for j in range(len(self._majority)):
                    self._distances[i][j] = distance(self._centers[i], self._majority[j])

        else:
            self._distances = np.zeros((len(self._minority), len(self._majority)))

            for i in range(len(self._minority)):
                for j in range(len(self._majority)):
                    self._distances[i][j] = distance(self._minority[i], self._majority[j])

    def fit_sample(self, X, y):
        self._X = X
        self._y = y
        self.set_distances()

        self.fracs = self.fracs/sum(self.fracs)

        majority = np.copy(self._majority)
        minority = np.copy(self._minority)

        if self._centers is not None:
            center_points = np.copy(self._centers)
        else:
            center_points = minority

        majority_to_delete = []

        for i in range(len(center_points)):
            sorted_distances = np.argsort(self._distances[i])
            j = 0
            try:
                while j < len(sorted_distances) and self._distances[i][sorted_distances[j]] < self.rads[i]:
                    majority_to_delete.append(sorted_distances[j])
                    j += 1
            except:
                print('lalalala')

        majority_to_delete = list(set(majority_to_delete))

        self.deleted_samples = majority[majority_to_delete]
        majority = np.delete(majority, majority_to_delete, axis=0)

        n = len(majority) - len(minority)

        self.appended = []
        self.samples = []
        for i in range(len(center_points)):
            minority_point = center_points[i]
            synthetic_samples = int(self.fracs[i] * n)
            self.samples.append(synthetic_samples)
            r = self.rads[i]

            for _ in range(synthetic_samples):
                self.appended.append(minority_point + taxicab_sample(len(minority_point), r))

        self.appended = np.array(self.appended)

        if len(self.appended) == 0:
            return np.concatenate([majority, minority]), \
               np.concatenate([np.full(len(majority), self.majority_class),
                               np.full( len(minority), self.minority_class)])

        return np.concatenate([majority, minority, self.appended]), \
               np.concatenate([np.full(len(majority), self.majority_class),
                               np.full( len(minority) + len(self.appended), self.minority_class)])


class PymooProblemWithCV(ElementwiseProblem):
    def __init__(self, n_min, classifier, X, y, centers, measures):
        self.n_var = 2*n_min
        self.measures = measures
        self.classifier = classifier
        self.X = X
        self.y = y
        self.centers = centers
        super().__init__(n_var=self.n_var,
                         n_obj=len(measures),
                         n_constr=0,
                         xl=np.full((self.n_var,), 0.0),
                         xu=np.full((self.n_var,), 1.0),
                         type_var=float)

    def _evaluate(self, x, out, *args, **kwargs):
        rads = x[0:int(self.n_var/2)]
        fracs = x[int(self.n_var/2):]

        sss = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0)
        measures_values = []
        sampler = HybridCenterSampling(rads, fracs, self.centers)
        for train_index, test_index in sss.split(self.X, self.y):
            new_X, new_y = sampler.fit_sample(self.X[train_index], self.y[train_index])
            c = clone(self.classifier)
            c.fit(new_X, new_y)
            y_pred = c.predict(self.X[test_index])
            measures_values.append([-m(self.y[test_index], y_pred) for m in self.measures])
        out["F"] = np.mean(np.array(measures_values), axis=0)


class MooHcsSelection:
    def __init__(self, classifier, measures, criteria=['best']):
        self.classifier = classifier
        self.measures = measures
        self.criteria = criteria

    def pick_solutions(self, results, criteria):
        def pick_best(solutions, objectives, index):
            return solutions[objectives.index(max(objectives, key=itemgetter(index)))]

        def pick_balanced(solutions, objectives):
            return solutions[objectives.index(min(objectives, key=lambda i: abs(i[0] - i[1])))]

        solutions = results.X
        objectives = [tuple(-obj) for obj in results.F]

        picked_solutions = []

        for criterion in criteria:
            if criterion == 'best':
                for i in range(len(objectives[0])):
                    picked_solutions.append(pick_best(solutions, objectives, i))
            elif criterion == 'balanced':
                picked_solutions.append(pick_balanced(solutions, objectives))
        return picked_solutions

    def fit_sample(self, X, y):

        classes = np.unique(y)

        X = X.astype('float32')
        y = y.astype('float32')

        sizes = [sum(y == c) for c in classes]

        minority = X[y == classes[np.argmin(sizes)]]

        clustering = KMeans(len(minority)//3).fit(X[y == classes[np.argmin(sizes)]])
        centers = clustering.cluster_centers_

        problem = PymooProblemWithCV(len(centers), self.classifier, X, y, centers, self.measures)

        algorithm = NSGA2(
            pop_size=400,
            sampling=FloatRandomSampling(),
            crossover=UniformCrossover(),
            mutation=GaussianMutation(),
            eliminate_duplicates=True
        )

        termination = MaximumGenerationTermination(1000)
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=False)

        solutions = self.pick_solutions(res, self.criteria)
        results = []

        for x in solutions:
            sampler = HybridCenterSampling(x[0:len(x)//2], x[len(x)//2:], centers)
            results.append(sampler.fit_sample(X,y))

        return results



