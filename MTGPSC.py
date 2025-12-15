import numpy as np
import math
import copy
import random
import operator
from deap import base, creator, gp, tools
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.spatial.distance import cityblock
from gp2 import SSC, staticLimit_multi, SSC_mate, quick_evaluate_subtree
from functools import partial


# test upload by github
class GPIndividualEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, pset, individual):
        self.pset = pset
        self.individual = individual
        self.func = gp.compile(self.individual, self.pset)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([self.func(*x) for x in X])


class SymbolicRegressorGP(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible class for symbolic regression using DEAP."""

    def __init__(
        self,
        n_generations=100,
        pop_size=100,
        crossover_prob=0.9,
        mutation_prob=0.1,
        n_selected_features=None,
        verbose=False,
        tr=0.3,
        semantic_crossover=True,
    ):
        self.n_generations = n_generations
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_selected_features = n_selected_features
        self.verbose = verbose
        self.toolbox = base.Toolbox()
        self.tr = tr
        self.semantic_crossover = semantic_crossover

    def SSD(self, parent_a, parent_b):
        func_a = gp.compile(parent_a, self.pset)
        func_b = gp.compile(parent_b, self.pset)
        y_pred_a = []
        y_pred_b = []
        for row in self.X:
            try:
                y_pred_a.append(func_a(*row))
            except:
                y_pred_a.append(np.nan)

        for row in self.X:
            try:
                y_pred_b.append(func_b(*row))
            except:
                y_pred_b.append(np.nan)

        Distance = cityblock(y_pred_a, y_pred_b) / len(self.X)
        return Distance

    def eaSimple_Elitism(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
    ):
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        self.sub_population = [0 for k in range(self.y.shape[1])]
        offspring = [0 for k in range(self.y.shape[1])]
        self.Best_of_individual = [0 for k in range(self.y.shape[1])]
        # split a population into k subpopulations
        for k in range(self.y.shape[1]):
            self.sub_population[k] = []
            offspring[k] = []
            self.Best_of_individual[k] = []
        # Evaluate the individuals with an invalid fitness
        j = 0
        for ind in zip(invalid_ind):
            t = math.floor(j / math.floor(self.pop_size))
            if t == self.y.shape[1]:
                t = self.y.shape[1] - 1
            ind[0].fitness.values = self.evaluate_individual(ind[0], t)
            self.sub_population[t].append(ind[0])
            j = j + 1

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for i in range(self.y.shape[1]):
            top = tools.selBest(self.sub_population[i], k=1)
            # Elitism[i] = top[0]
        # Begin the generational process
        self.redunt_time = 0
        for gen in range(1, ngen + 1):
            # # Select the next generation individuals based on Elitism
            new_population = []
            len_invalid = 0
            for task in range(self.y.shape[1]):
                len_sub_population = self.pop_size
                offspring_selected = toolbox.select(
                    self.sub_population[task], len_sub_population
                )
                clone_offspring = copy.deepcopy(offspring_selected)
                clone_offspring_mut = copy.deepcopy(offspring_selected)

                offspring[task] = []
                # crossover
                for j in range(int(len_sub_population)):
                    if len(offspring[task]) >= 100:
                        break
                    if random.random() < cxpb:
                        # crossover in different tasks
                        if random.random() <= self.tr:
                            parent_a = toolbox.select(self.sub_population[task], 1)
                            parent_a_copy = copy.deepcopy(parent_a)
                            # parent_a_copy = copy.deepcopy(parent_a)
                            while True:
                                random_t = random.randint(0, self.y.shape[1] - 1)
                                if random_t != task:
                                    break
                            parent_b = toolbox.select(self.sub_population[random_t], 1)
                            parent_b_copy = copy.deepcopy(parent_b)

                            # semantic similarity in crossover
                            child_a, child_b = toolbox.semantic(
                                parent_a_copy[0], parent_b_copy[0], self.y[:, task]
                            )
                            # self.redunt_time = time + self.redunt_time

                            del child_a.fitness.values, child_b.fitness.values
                            offspring[task].append(child_a)
                            # offspring[task].append(child_b)

                        # crossover in the same task
                        else:
                            parent_a = copy.deepcopy(clone_offspring[j])
                            try:
                                parent_b = copy.deepcopy(clone_offspring[j + 1])
                                j = j + 1
                            except:
                                continue
                            child_a, child_b = toolbox.mate(parent_a, parent_b)
                            # self.redunt_time = time + self.redunt_time
                            del child_a.fitness.values, child_b.fitness.values
                            offspring[task].append(child_a)
                            offspring[task].append(child_b)

                    else:
                        parent = toolbox.select(clone_offspring_mut, 1)
                        selected_parent = copy.deepcopy(parent)
                        if random.random() < mutpb:
                            (selected_parent,) = toolbox.mutate(selected_parent[0])
                            del selected_parent.fitness.values
                            offspring[task].append(selected_parent)

                invalid_ind = [ind for ind in offspring[task] if not ind.fitness.valid]

                seen_individuals = set()
                new_unique_ind = []
                for ind in zip(invalid_ind):
                    ind_str = str(ind[0])
                    if ind_str not in seen_individuals:
                        # 只评估不重复的个体
                        seen_individuals.add(ind_str)
                        ind[0].fitness.values = self.evaluate_individual(ind[0], task)
                        # 只保留不重复的个体
                        new_unique_ind.append(ind[0])
                        len_invalid = len_invalid + 1
                offspring[task] = new_unique_ind



                # Elitism
                elitism = tools.selBest(self.sub_population[task], k=1)
                for i in range(len(elitism)):
                    offspring[task].append(elitism[i])

                if int(len_sub_population - len(offspring[task])) > 0:
                    rest = self.toolbox.population(n=int(len_sub_population - len(offspring[task])))
                    for i in range(int(len_sub_population - len(offspring[task]))):
                        rest[i].fitness.values = self.evaluate_individual(rest[i], task)
                        len_invalid = len_invalid + 1
                        offspring[task].append(rest[i])
                self.sub_population[task] = offspring[task]
                for i in range(len(self.sub_population[task])):
                    new_population.append(self.sub_population[task][i])

            # Append the current generation statistics to the logbook
            record = stats.compile(new_population) if stats else {}
            logbook.record(gen=gen, nevals=len_invalid, **record)
            if verbose:
                print(logbook.stream)

            # Store best-of-individual in every generation
            for i in range(self.y.shape[1]):
                top = tools.selBest(self.sub_population[i], k=1)
                self.Best_of_individual[i].append(top[0])

        return new_population, logbook

    def fit(self, X, y):
        self.X = X
        self.y = y
        # Define primitive set
        pset = gp.PrimitiveSet("MAIN", X.shape[1])
        self.add_functions_to_pset(pset)
        pset.addEphemeralConstant("rand101", lambda: np.random.randint(-1, 1))
        self.pset = pset

        # Define individual and population
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        class PrimitiveTreeCopy(gp.PrimitiveTree):
            def __deepcopy__(self, memo):
                new_content = copy.deepcopy(self[:], memo)
                new_instance = self.__class__(new_content)
                copied_dict = {
                    key: (
                        value
                        if key in ("semantics", "subtree_semantics")
                        else copy.deepcopy(value, memo)
                    )
                    for key, value in self.__dict__.items()
                }
                new_instance.__dict__.update(copied_dict)
                return new_instance

        creator.create("Individual", PrimitiveTreeCopy, fitness=creator.FitnessMin)
        self.define_toolbox()

        # Initialize population
        pop = self.toolbox.population(n=self.pop_size * self.y.shape[1])

        # Define the statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: ind.height)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        # Store top 10% individuals
        hof = tools.HallOfFame(1)

        # Run GP algorithm for full set of features
        # algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
        #                     ngen=self.n_generations, stats=mstats, halloffame=hof, verbose=self.verbose)
        self.eaSimple_Elitism(
            pop,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=mstats,
            halloffame=hof,
            verbose=self.verbose,
        )

        # Obtain the best model based on standard GP
        self.top = [0 for k in range(self.y.shape[1])]
        self.final_model = [0 for k in range(self.y.shape[1])]
        for i in range(self.y.shape[1]):
            self.top[i] = tools.selBest(self.sub_population[i], k=1)
            self.final_model[i] = self.top[i][0]
        # self.func_standard = gp.compile(hof[0], self.pset)
        # top = tools.selBest(hof, k=1)
        # self.func_standard = gp.compile(top[0], self.pset)
        # self.top=top

        return self

    def evaluate_individual(self, individual, t):
        y_pred, y_subtree_semantics = quick_evaluate_subtree(
            individual, self.pset, self.X
        )
        individual.subtree_semantics = {k: v for k, v in y_subtree_semantics}

        return (
            math.sqrt(np.mean((self.y[:, t] - y_pred) ** 2))
            / (self.y[:, t].max() - self.y[:, t].min()),
        )

    def add_functions_to_pset(self, pset):
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        # pset.addPrimitive(self.protectedDiv, 2)
        pset.addPrimitive(self.sqrt, 1)
        pset.addPrimitive(self.inv, 1)

    def inv(self, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x) > 0, np.divide(1, x), 1)

    def protectedDiv(self, x1, x2):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x2) > 0, np.divide(x1, x2), 1)

    def sqrt(self, x):
        return np.sqrt(np.abs(x))

    def define_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=6)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        # self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("semantic", partial(SSC, pset=self.pset, X=self.X))
        if self.semantic_crossover:
            self.toolbox.register("mate", partial(SSC_mate, pset=self.pset, X=self.X))
        else:
            self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=10)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.register("select", tools.selTournament, tournsize=7)
        # self.toolbox.decorate(
        #     "mate", staticLimit_multi(key=operator.attrgetter("height"), max_value=10)
        # )
        # self.toolbox.decorate(
        #     "semantic",
        #     staticLimit_multi(key=operator.attrgetter("height"), max_value=10),
        # )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10)
        )

    def predict_Standard(self, X, func):
        y_pred = []
        for row in X:
            try:
                y_pred.append(func(*row))
            except:
                y_pred.append(np.nan)
        return y_pred
