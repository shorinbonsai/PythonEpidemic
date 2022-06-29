from cProfile import run
import sys
import os
import csv
import math
import random
import re
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import copy

from PIL import Image, ImageTk
from graphviz import Graph
from statistics import mean
from math import exp, log

from numpy.random import randint
from numpy.random import rand

sns.set_theme(style="ticks", color_codes=True)


alpha = 0.3
# shutdown = 3
chance_remove = 0.5
shutdown_percent = 0.05
reopen_percent = 0.02


def infected(sick: int):
    beta = 1 - exp(sick * log(1 - alpha))
    if random.random() < beta:
        return True
    return False


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



def run_epis(adj_lists, nodes, p0: int = 2, remove_list: list[list[int]] = [], edge_list: list = [], output: str = ""):
    epidemics = 50
    epi_logs = []
    lengths = []
    sums = [0 for _ in range(nodes)]
    counts = [0 for _ in range(nodes)]
    totals = []
    stops = []
    reopens = []
    for _ in range(epidemics):
        epi_log, temp, stop, reopen = mod_reopen(
            adj_lists, nodes, p0, remove_list)
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

    return epi_logs, avg, avg_all
    

    # x = [n for n in range(max_len)]
    # x_lbls = [str(n) for n in range(max_len)]

    # fig = matplotlib.pyplot.figure()
    # fig.set_dpi(400)
    # fig.set_figheight(4)
    # plot = fig.add_subplot(111)

    # for el in epi_logs:
    #     plot.plot(x, el, linewidth=1, alpha=0.3, color='gray')
    #     pass

    # plot.plot(x, avg, label="Average of Running")
    # plot.plot(x, avg_all, label="Average of All")
    # fig.suptitle(
    #     "Epidemic Profiles for 50 Epidemics")
    # plot.set_ylabel("Newly Infected Individuals")
    # plot.set_xlabel("Day \n DATA:[Avg Lockdown: " +
    #                 str(avg_stops) + " Avg Infected: " + str(avg_total) + " Avg Reopen: " + str(avg_reopen) + "]")
    # plot.set_xticks(x)
    # plot.set_xticklabels(x_lbls)
    # plot.legend()
    # fig.tight_layout()
    # fig.savefig(output + str(seed) + "_epi.png")


def main():
    p0 = 0

    # graph_inp = sys.argv[1]
    # lockgraph = sys.argv[2]
    strict50 = []
    strict70 = []
    medium50 = []
    medium70 = []
    lax50 = []
    lax70 = []
    skeptical50 = []
    skeptical70 = []
    strictness = 0
    stuff50 = [[] for _ in range(4)]
    stuff70 = [[] for _ in range(4)]
    graph_inp = ""
    for dirpath, dirnames, files in os.walk('.'):
        for file_name in files:
            if file_name.endswith("graph.dat"):
                direc = os.path.join(dirpath, file_name)

                if file_name.__contains__("256N"):
                    p0 = 10
                    graph_inp = "256N_graph10.dat"
                elif file_name.__contains__("512N"):
                    p0 = 3
                    graph_inp = "512N_graph3.dat"
                elif file_name.__contains__("768"):
                    p0 = 5
                    graph_inp = "768N_graph5.dat"
                elif file_name.__contains__("1024"):
                    p0 = 18
                    graph_inp = "1024N_graph18.dat"
                if file_name.__contains__("strict"):
                    strictness = 0
                elif file_name.__contains__("medium"):
                    strictness = 1
                elif file_name.__contains__("lax"):
                    strictness = 2
                elif file_name.__contains__("leery"):
                    strictness = 3
                if file_name.__contains__("rem50"):
                    removal = 0
                elif file_name.__contains__("rem70"):
                    removal = 1   

                orig_adjlist = make_adj_lists(graph_inp)
                node_numb = len(orig_adjlist)
                lockdown_adjlist = make_adj_lists(direc)
                _, _, epidem = run_epis(orig_adjlist, node_numb, p0, lockdown_adjlist)

                if removal == 0:
                    stuff50[strictness].append(epidem)
                elif removal == 1:
                    stuff70[strictness].append(epidem)
            
                              


    # edgelist = get_edge_list(graph_inp)
    # node_numb, edges_numb = edgelist.pop(0)
    # orig_adjlist = make_adj_lists(graph_inp)
    # node_numb = len(orig_adjlist)
    # lockdown_adjlist = make_adj_lists(lockgraph)

    # test_epis, avg, avg_all = run_epis(orig_adjlist, node_numb,p0, lockdown_adjlist)
    print("cat")
    return


if '__main__' == __name__:
    main()