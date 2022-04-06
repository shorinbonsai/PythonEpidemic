import copy
import operator
import math
import random
import sys
import numpy
import pandas as pd
from math import exp, log

from numpy.random import randint
from numpy.random import rand

from typing import List, Set, Dict, Tuple, Optional

seed = 4587955
alpha = 0.3
shutdown = 3
chance_remove = 0.5
shutdown_percent = 0.20
reopen_percent = 0.05
MAX_GENERATIONS = 50
POP_SIZE = 100
CROSSOVER_RATE = 0.75
MUT_RATE = 0.25


# legacy code begin


def infected(sick: int):
    beta = 1 - exp(sick * log(1 - alpha))
    if random.random() < beta:
        return True
    return False


def fitness_reopen(adj_lists: list[int], nodes: int, p0, remove_list: list[int] = [], edge_list: list = []):
    temp_list = copy.deepcopy(adj_lists)
    n_state = [0 for _ in range(nodes)]  # susceptible
    n_state[p0] = 1
    epi_log = [[p0]]
    num_infected = 1
    ttl_infected = 0
    time_step = 0
    # alpha = 0.3
    have_locked_down = False
    have_reopened = False
    lockdown_step = 0
    reopen_step = 128
    length = 0
    while num_infected > 0:
        current_infected = num_infected/nodes
        if current_infected >= shutdown_percent and have_locked_down == False:
            # lockdown
            for idx, edge in enumerate(remove_list):
                if edge == 1:
                    x = edge_list[idx][0]
                    y = edge_list[idx][1]
                    temp_list = remove_edge_adj(x, y, temp_list)

            have_locked_down = True
            lockdown_step = time_step

        inf_neighbours = [0 for _ in range(nodes)]

        # if threshold met then restore initial contact graph
        current_infected = num_infected/nodes
        if current_infected < reopen_percent and have_locked_down == True and have_reopened == False:
            temp_list = copy.deepcopy(adj_lists)
            reopen_step = time_step
            have_reopened = True

        for n in range(nodes):
            if n_state[n] == 1:
                for nei in temp_list[n]:
                    inf_neighbours[nei] += 1
                    pass
                pass
            pass

        for n in range(nodes):
            if n_state[n] == 0 and inf_neighbours[n] > 0:
                if infected(inf_neighbours[n]):
                    n_state[n] = 3
                    pass
                pass
            pass

        ttl_infected += num_infected
        num_infected = 0
        new_inf = []
        for n in range(nodes):
            if n_state[n] == 1:  # infected -> removed
                n_state[n] = 2
                pass
            elif n_state[n] == 3:
                n_state[n] = 1
                num_infected += 1
                new_inf.append(n)
                pass
            pass
        epi_log.append(new_inf)
        length += 1
        time_step += 1
        pass
    # return epi_log, ttl_infected, lockdown_step, reopen_step
    return ttl_infected


def fitness_lockdown(adj_lists: list, nodes: int, p0, remove_list: list[int] = [], edge_list: list = []):
    temp_list = copy.deepcopy(adj_lists)
    n_state = [0 for _ in range(nodes)]  # susceptible
    n_state[p0] = 1
    epi_log = [[p0]]
    num_infected = 1
    ttl_infected = 0
    time_step = 0
    # alpha = 0.3
    have_locked_down = False
    lockdown_step = 0
    length = 0
    while num_infected > 0:
        if num_infected/nodes >= shutdown_percent and have_locked_down == False:
            # lockdown
            for idx, edge in enumerate(remove_list):
                if edge == 1:
                    x = edge_list[idx][0]
                    y = edge_list[idx][1]
                    temp_list = remove_edge_adj(x, y, temp_list)

            have_locked_down = True
            lockdown_step = time_step

        inf_neighbours = [0 for _ in range(nodes)]

        for n in range(nodes):
            if n_state[n] == 1:
                for nei in temp_list[n]:
                    inf_neighbours[nei] += 1
                    pass
                pass
            pass

        for n in range(nodes):
            if n_state[n] == 0 and inf_neighbours[n] > 0:
                if infected(inf_neighbours[n]):
                    n_state[n] = 3
                    pass
                pass
            pass

        ttl_infected += num_infected
        num_infected = 0
        new_inf = []
        for n in range(nodes):
            if n_state[n] == 1:  # infected -> removed
                n_state[n] = 2
                pass
            elif n_state[n] == 3:
                n_state[n] = 1
                num_infected += 1
                new_inf.append(n)
                pass
            pass
        epi_log.append(new_inf)
        length += 1
        time_step += 1
        pass
    # return epi_log, ttl_infected, lockdown_step
    return ttl_infected


# old code end


def get_edge_list(inp: str):
    with open(inp, "r") as f:
        first_line = f.readline()
        nodes = int(first_line.rstrip().split('\t')[0].split(' ')[1])
        edg_list = [nodes]
        lines = f.readlines()
        for fr, line in enumerate(lines):
            line = line.rstrip()
            line = line.split(' ')
            if len(line) > 0:
                for to in line:
                    if to != '':
                        edg_list.append((fr, int(to)))
                        pass
                    pass
                pass
            pass
    return edg_list


def make_adj_lists(inp: str):
    with open(inp, "r") as f:
        first_line = f.readline()
        nodes = int(first_line.rstrip().split('\t')[0].split(' ')[1])
        adj_lists = [[] for _ in range(nodes)]
        lines = f.readlines()
        for fr, line in enumerate(lines):
            line = line.rstrip()
            line = line.split(' ')
            if len(line) > 0:
                for to in line:
                    if to != '':
                        adj_lists[fr].append(int(to))
                        pass
                    pass
                pass
            pass

    return adj_lists


def init_pop(edge_count: int, percent_lockdown: float, pop_size: int) -> list[int]:
    # total_edges = list(range(edge_count))
    total_edges = [1 for i in range(edge_count)]
    pop = []
    for i in range(pop_size):
        new_ind = total_edges.copy()
        numb_remove = int(percent_lockdown * edge_count)
        remove_list = random.sample(list(enumerate(new_ind)), numb_remove)
        for j in remove_list:
            new_ind[j[0]] = 0

        pop.append(new_ind)
    return pop

# safe dealer based crossover


def sdb(individual1: list[int], individual2: list[int], cross_chance: float) -> tuple[list, list]:
    c1, c2 = copy.deepcopy(individual1), copy.deepcopy(individual2)
    # check if cross
    if random.random() < cross_chance:
        ind_common = []
        ind1_ones = []
        ind2_ones = []
        length = len(individual1)
        # get set intersections
        for indx, value in enumerate(individual1):
            if individual2[indx] == value and value == 1:
                ind_common.append(indx)
            elif value == 1:
                ind1_ones.append(indx)
        for indx, value in enumerate(individual2):
            if value == 1 and indx not in ind_common:
                ind2_ones.append(indx)
        ind1_ones.extend(ind2_ones)
        ind1_ones_set = set(ind1_ones)
        # split set of 1's into two children
        child1 = set(random.sample(
            ind1_ones_set, math.floor(len(ind1_ones_set)/2)))
        child2 = ind1_ones_set - child1
        child1 = child1.union(ind_common)
        child2 = child2.union(ind_common)
        # new children chromosomes
        c1 = [0] * length
        c2 = [0] * length
        for i in child1:
            c1[i] = 1
        for i in child2:
            c2[i] = 1
    return c1, c2


'''index swap mutation, will swap from 1 - 5 pairs of indices'''


def mutate(indiv: list[int], chance_mut: float) -> list[int]:
    result = copy.deepcopy(indiv)
    if random.random() < chance_mut:
        numb = random.randint(1, 5)
        for _ in range(numb):
            idx = random.choice(range(len(result)))
            idx2 = random.choice(range(len(result)))
            if result[idx] == 0 and result[idx2] == 1:
                result[idx] = 1
                result[idx2] == 0
            elif result[idx] == 1 and result[idx2] == 0:
                result[idx] = 0
                result[idx2] = 1
            elif result[idx] == 1 and result[idx2] == 1:
                continue
            elif result[idx] == 0 and result[idx2] == 0:
                continue
    return result

# class Graph:
#     edgelist = []

#     def __init__(self, edgelist):
#         self.edgelist = edgelist

#     def remove_edge(self, v1, v2):
#         for i in self.edgelist:
#             if i[0] == v1 and i[1] == v2:
#                 self.edgelist.remove(i)
#                 break
#         for i in self.edgelist:
#             if i[0] == v2 and i[1] == v1:
#                 self.edgelist.remove(i)
#                 break


def remove_edge(v1, v2, edgelist):
    for i in edgelist:
        if i[0] == v1 and i[1] == v2:
            edgelist.remove(i)
            break
    for i in edgelist:
        if i[0] == v2 and i[1] == v1:
            edgelist.remove(i)
            break
    return edgelist


def remove_edge_adj(v1: int, v2: int, adjlist: list[list[int]]) -> list[list[int]]:
    workinglist = copy.deepcopy(adjlist)
    try:
        workinglist[v1].remove(v2)
    except ValueError:
        pass
    try:
        workinglist[v2].remove(v1)
    except ValueError:
        pass
    return workinglist

# tournament selection


def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def main():
    # random.seed(seed)
    edgelist = get_edge_list("256N_graph0.dat")
    node_numb = edgelist.pop(0)
    adjlist = make_adj_lists("256N_graph0.dat")

    # GA logic
    pop = init_pop(node_numb, chance_remove, POP_SIZE)
    # evaluate all candidates in the population
    p0 = 19
    best_eval = 99999
    scores = []
    for gen in range(MAX_GENERATIONS):
        # fitness

        for i in pop:
            tmp = 0
            for j in range(10):

                fit = 99999
                while fit >= 9999:
                    score = fitness_reopen(
                        adjlist, node_numb, p0, i, edgelist)
                    if score >= 5:
                        tmp = tmp + score
                        fit = score
            fitness = score/10
            scores.append(fitness)

        # scores = [fitness_lockdown(
        #     adjlist, node_numb, p0, individual, edgelist) for individual in pop]
        # find best individual
        for i in range(POP_SIZE):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print("gen >%d, new best = %.3f" % (gen, scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(POP_SIZE)]
        children = list()
        for i in range(0, POP_SIZE, 2):
            # pairs of parents
            p1, p2 = selected[i], selected[i+1]
            # cross/mut
            for c in sdb(p1, p2, CROSSOVER_RATE):
                # mutate
                tmp = mutate(c, MUT_RATE)
                children.append(tmp)
        pop = children
        print("best: " + str(best_eval) + " gen: " + str(gen))

    cat = get_edge_list("256N_graph0.dat")
    num_nodes = cat.remove(cat[0])
    print(cat)
    cat = remove_edge(0, 157, cat)
    print("cat")


if __name__ == "__main__":
    main()
