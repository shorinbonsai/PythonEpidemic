import copy
import operator
import math
import random
import sys
import numpy
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

graph_inp = "512N_graph3.dat"
POP_SIZE = 101
chance_remove = 0.7


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


def main():
    edgelist = get_edge_list(graph_inp)
    node_numb, edges_numb = edgelist.pop(0)
    adjlist = make_adj_lists(graph_inp)

    pop = init_pop(edges_numb, chance_remove, POP_SIZE, edgelist)

    graphy = make_disconnected_graph(adjlist, pop[5], edgelist)

    count = 0
    for i in graphy:
        count += len(i)
    count2 = pop[5].count(1)

    with open("512a_randomgraph.dat", 'w') as f:
        print("nodes: " + str(node_numb) +
              " edges: " + str(int(count2)), file=f)
        for i in graphy:
            for j in i:
                print(str(j) + " ", end="", file=f)
            print("", file=f)


    return



if __name__ == "__main__":
    main()