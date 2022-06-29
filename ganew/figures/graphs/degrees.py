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
from statistics import mean, median

from numpy.random import randint
from numpy.random import rand

from typing import List, Set, Dict, Tuple, Optional





def make_adj_lists(inp: str):
    with open(inp, "r") as f:
        first_line = f.readline()
        # nodes = int(first_line.rstrip().split('\t')[0].split(' ')[1])
        adj_lists = [[] for _ in range(512)]
        lines = f.readlines()
        for fr, line in enumerate(lines):
            line = line.rstrip()
            line = line.split(' ')
            for to in line:
                if to != '':
                    adj_lists[fr].append(int(to))
                    pass
                pass
            pass

    return adj_lists

def get_degrees(inp: list):
    result = []
    for val in inp:
        result.append(len(set(val)))
    return result



def main():
    graph_inp = sys.argv[1]
    adj = make_adj_lists(graph_inp)
    degrees = get_degrees(adj)
    avg_deg = mean(degrees)
    med_deg = median(degrees)
    print(avg_deg)

if __name__ == "__main__":
    main()
