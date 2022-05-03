import copy
import operator
import math
import random
import sys
import numpy
# import pandas as pd
from math import exp, log
import matplotlib.pyplot
from PIL import Image, ImageTk
from graphviz import Graph
from statistics import mean

from numpy.random import randint
from numpy.random import rand

from typing import List, Set, Dict, Tuple, Optional

# seed = 4587955
alpha = 0.3
shutdown = 3
# chance_remove = 0.5
# shutdown_percent = 0.05
# reopen_percent = 0.02
MAX_GENERATIONS = 20
# POP_SIZE = 101
# CROSSOVER_RATE = 0.50
# MUT_RATE = 0.50

p0 = int(sys.argv[1])
graph_inp = sys.argv[2]
shutdown_percent = float(sys.argv[3])
reopen_percent = float(sys.argv[4])
chance_remove = float(sys.argv[5])
CROSSOVER_RATE = float(sys.argv[6])
MUT_RATE = float(sys.argv[7])
POP_SIZE = int(sys.argv[8])
output = sys.argv[9]
seed = int(sys.argv[10])


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
                if edge == 0:
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
    return epi_log, ttl_infected, lockdown_step, reopen_step
    # return ttl_infected


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
        first_line = first_line.strip('\n')
        first_line = first_line.split(' ')
        nodes = int(first_line[0])
        edges = int(first_line[2])
        edg_list = [(nodes, edges)]
        lines = f.readlines()
        for fr, line in enumerate(lines):
            line = line.rstrip()
            line = line.split(' ')
            if len(line) > 0:
                for to in line:
                    if to != '' and int(to) > fr:
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


def check_dup(testlist):
    for i in testlist:
        if (testlist.count(i) > 1):
            return True
    return False


def init_pop(edge_count: int, percent_lockdown: float, pop_size: int, edg_list: list[(int, int)]) -> list[int]:
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

        # debug tests
        # test = False
        # if check_dup(rem_list):
        #     test = True
        # for j in remove_list:
        #     new_ind[j[0]] = 0

        pop.append(new_ind)
        # debug tests
        zeros = new_ind.count(0)
        ones = new_ind.count(1)
        # zeros = new_ind.count(0)
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


def get_second_idx(idx: int, edg_list: list[(int, int)]) -> list[int]:
    second_val = edg_list[idx][1]
    candidates = []
    for i in range(len(edg_list)):
        if edg_list[i][0] == second_val and edg_list[i][1] == edg_list[idx][0]:
            candidates.append(i)
    # second_idx = random.choice(candidates)
    return candidates


'''index swap mutation, will swap 2 pairs of indices'''


def mutate(indiv: list[int], chance_mut: float, edg_list: list[(int, int)]) -> list[int]:
    result = copy.deepcopy(indiv)
    if random.random() < chance_mut:
        # numb = random.randint(1, 5)
        count = 0
        while count < 2:
            idx = random.choice(range(len(result)))
            idx2 = random.choice(range(len(result)))
            if result[idx] == 0 and result[idx2] == 1:
                result[idx] = 1
                result[idx2] == 0

                count += 1
            elif result[idx] == 1 and result[idx2] == 0:
                result[idx] = 0
                result[idx2] = 1

                count += 1
            elif result[idx] == 1 and result[idx2] == 1:
                continue
            elif result[idx] == 0 and result[idx2] == 0:
                continue
    return result


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
    workinglist = adjlist
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


def selection(pop, scores, k=5):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def make_graphs(inp: list, epi_log: list):
    removed = []
    for num, li in enumerate(epi_log):
        make_graph(inp, "graph" + str(num), li, removed)
        for n in li:
            removed.append(n)
            pass
        pass
    pass


def make_graph(inp: list, out: str, inf: list, rem: list):
    g = Graph(engine='sfdp')
    g.attr(size="6,6")
    g.graph_attr.update(dpi='600', overlap='false')
    g.node_attr.update(shape='circle', style='filled',
                       fontsize='12', fixedsize='true')
    g.edge_attr.update()

    for n in range(inp[0]):
        if n in inf:
            g.node(str(n), label=str(n), fillcolor='red')
            pass
        elif n in rem:
            g.node(str(n), label=str(n), fillcolor='green')
            pass
        else:
            g.node(str(n), label=str(n), fillcolor='white')
            pass
        pass

    for e in inp[1:]:
        if e[0] >= e[1]:
            g.edge(str(e[0]), str(e[1]))
            pass
        pass

    g.render(filename=out, cleanup=True, format='png')
    pass


def make_disconnected_graph(adjlist: list, remove_list: list, edgelist: list) -> list:
    temp_list = copy.deepcopy(adjlist)
    count2 = remove_list.count(1)
    for idx, edge in enumerate(remove_list):
        if edge == 0:
            x = edgelist[idx][0]
            y = edgelist[idx][1]
            sumy = 0
            for i in temp_list:
                sumy += len(i)
            temp_list = remove_edge_adj(x, y, temp_list)
    return temp_list


def run_epis(adj_lists, nodes, p0: int = 2, remove_list: list[int] = [], edge_list: list = [], output: str = ""):
    epidemics = 50
    epi_logs = []
    lengths = []
    sums = [0 for _ in range(nodes)]
    counts = [0 for _ in range(nodes)]
    totals = []
    stops = []
    reopens = []
    for _ in range(epidemics):
        epi_log, temp, stop, reopen = fitness_reopen(
            adj_lists, nodes, p0, remove_list, edge_list)
        epi_log = [len(n) for n in epi_log]
        epi_logs.append(epi_log)
        lengths.append(len(epi_log))
        totals.append(temp)
        stops.append(stop)
        if reopen <= len(epi_log):
            reopens.append(reopen)
        pass
    avg_total = mean(totals)

    if mean(stops) > 0:
        avg_stops = mean(stops)
    else:
        avg_stops = "N/A"

    if len(reopens) > 0:
        avg_reopen = mean(reopens)
    else:
        avg_reopen = "N/A"
    print(reopens)

    for ln, el in enumerate(epi_logs):
        for day in range(lengths[ln]):
            sums[day] += el[day]
            counts[day] += 1
            pass
        pass

    avg = []
    avg_all = []
    for day, s in enumerate(sums):
        if counts[day] > 0:
            avg.append(s / counts[day])
            avg_all.append(s / epidemics)
        pass

    max_len = max(lengths)
    for el in epi_logs:
        s_len = len(el)
        for _ in range(max_len - s_len):
            el.append(0)
    x = [n for n in range(max_len)]
    x_lbls = [str(n) for n in range(max_len)]

    fig = matplotlib.pyplot.figure()
    fig.set_dpi(400)
    fig.set_figheight(4)
    plot = fig.add_subplot(111)

    for el in epi_logs:
        plot.plot(x, el, linewidth=1, alpha=0.3, color='gray')
        pass

    plot.plot(x, avg, label="Average of Running")
    plot.plot(x, avg_all, label="Average of All")
    fig.suptitle(
        "Epidemic Profiles for 50 Epidemics")
    plot.set_ylabel("Newely Infected Individuals")
    plot.set_xlabel("Day \n DATA:[Avg Lockdown: " +
                    str(avg_stops) + " Avg Infected: " + str(avg_total) + " Avg Reopen: " + str(avg_reopen) + "]")
    plot.set_xticks(x)
    plot.set_xticklabels(x_lbls)
    plot.legend()
    fig.tight_layout()
    fig.savefig(output + str(seed) + "_epi.png")


def main():

    random.seed(seed)
    edgelist = get_edge_list(graph_inp)
    node_numb, edges_numb = edgelist.pop(0)
    adjlist = make_adj_lists(graph_inp)
    # output = "logfile.txt"

    # GA logic
    pop = init_pop(edges_numb, chance_remove, POP_SIZE, edgelist)
    numbones = pop[0].count(0)
    # evaluate all candidates in the population
    p0 = 19
    best_eval = 99999
    elite = []
    best = []
    logfile = output + str(seed) + "log.txt"
    with open(logfile, 'w') as f:

        for gen in range(MAX_GENERATIONS):
            # fitness
            scores = []
            for i in pop:
                tmp = 0
                # run 10 epidemics and only include those with length >= 5
                for j in range(10):

                    fit = 99999
                    while fit >= 9999:
                        _, score, _, _ = fitness_reopen(
                            adjlist, node_numb, p0, i, edgelist)
                        if score >= 5:
                            tmp = tmp + score
                            fit = score
                fitness = tmp/10
                scores.append(fitness)

            # find best individual
            for i in range(POP_SIZE):
                if scores[i] < best_eval:
                    elite, best_eval = pop[i], scores[i]
                    print("gen >%d, new best = %.3f" % (gen, scores[i]))
                    print("gen >%d, new best = %.3f" %
                          (gen, scores[i]), file=f)
            # select parents
            # selected = [selection(pop, scores, 7) for _ in range(POP_SIZE)]
            children = list()
            children.append(elite)
            for i in range(0, POP_SIZE-1, 2):
                # pairs of parents
                p1 = selection(pop, scores, 7)
                p2 = selection(pop, scores, 7)
                # p1, p2 = selected[i], selected[i+1]
                # cross/mut
                for c in sdb(p1, p2, CROSSOVER_RATE):
                    # mutate
                    tmp = mutate(c, MUT_RATE, edgelist)
                    children.append(tmp)
            pop = children
            print("gen: " + str(gen) + " best: " + str(best_eval) + " mean: " +
                  str(mean(scores)))
            print("gen: " + str(gen) + " best: " + str(best_eval) + " mean: " +
                  str(mean(scores)), file=f)

    run_epis(adjlist, node_numb, p0, elite, edgelist, output)
    discon = make_disconnected_graph(adjlist, elite, edgelist)
    count = 0
    for i in discon:
        count += len(i)
    count2 = elite.count(1)
    print("Test count of edges: " + str(count2))
    # ones = new_ind.count(1)
    graph_file = output + str(seed) + "graph.dat"
    with open(graph_file, 'w') as f:
        print("nodes: " + str(node_numb) +
              " edges: " + str(int(count2)), file=f)
        for i in discon:
            for j in i:
                print(str(j) + " ", end="", file=f)
            print("", file=f)

    print("cat")


if __name__ == "__main__":
    main()
