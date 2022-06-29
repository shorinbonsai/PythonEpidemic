import copy
import operator
import math
import random
import sys
import numpy as np
import csv
# import pandas as pd
from math import exp, log
import matplotlib.pyplot
from PIL import Image, ImageTk
from graphviz import Graph
from statistics import mean

from numpy.random import randint
from numpy.random import rand

from typing import List, Set, Dict, Tuple, Optional

alpha = 0.3
shutdown = 3


def get_bests(dir_path: str):
    with open(dir_path, "r",  errors="ignore") as f:
        best = f.readlines()
        result = []
        for i in best:
            i = i.strip()
            i = i.rstrip(".csv")
            i = i + "graph.dat"
            result.append(i)
        return result

def infected(sick: int):
    beta = 1 - exp(sick * log(1 - alpha))
    if random.random() < beta:
        return True
    return False





# def run_epidemics(orig: str, nodes, p0: int = 2, lockdown_graph: str):

#     return

shutdown_percent = 0.025
reopen_percent = 0.01

def main():
    random.seed(171329)
    p0 = 0
    global shutdown_percent
    global reopen_percent
    graph_inp = sys.argv[1]
    graph_orig = make_adj_lists(graph_inp)
    node_numb = len(graph_orig)
    

    test = get_bests("bestofbest.txt")
    if test[0].__contains__("graph10"):
        p0 = 10
    elif test[0].__contains__("graph19"):
        p0 = 19
    elif test[0].__contains__("graph22"):
        p0 = 22
    elif test[0].__contains__("graph3"):
        p0 = 3
    elif test[0].__contains__("graph9"):
        p0 = 9
    elif test[0].__contains__("graph18"):
        p0 = 18
    elif test[0].__contains__("graph5"):
        p0 = 5
    elif test[0].__contains__("graph21"):
        p0 = 21
    elif test[0].__contains__("graph24"):
        p0 = 24
    elif test[0].__contains__("graph26"):
        p0 = 26
    with open(graph_inp + "_stats2.txt", mode='w')  as f:
        print("Exp\t" + "Best\t" + "Mean\t" + "StdDev", file=f )
        for i in test:
            lockdown_graph = make_adj_lists(i)
            if i.__contains__("strict"):
                shutdown_percent = 0.025
                reopen_percent = 0.01
            elif i.__contains__("medium"):
                shutdown_percent = 0.05
                reopen_percent = 0.02
            elif i.__contains__("lax"):
                shutdown_percent = 0.075
                reopen_percent = 0.03
            elif i.__contains__("leery"):
                shutdown_percent = 0.1
                reopen_percent = 0.04
            result = []    
            for j in range(30):
                score = 0
                while score <5:
                    _, score, _, _ = mod_reopen(graph_orig,node_numb, p0,lockdown_graph)
                    result.append(score)
            print(i + "\t" + str(min(result)) + "\t" + str(mean(result)) + "\t" + str(np.std(result)), file=f)



    # for dirpath, dirnames, files in os.walk('.'):
    #     for file_name in files:
    #         if file_name.endswith(".csv"):
    #             direc = os.path.join(dirpath, file_name)
    print(test)

    return

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

def mod_reopen (adj_lists: list[int], nodes: int, p0, remove_list: list[int]):
    temp_list = copy.deepcopy(adj_lists)
    tmp_removed = copy.deepcopy(remove_list)
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
            temp_list = tmp_removed
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


if __name__ == "__main__":
    main()