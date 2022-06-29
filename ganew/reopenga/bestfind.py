import copy
import operator
import math
import random
import sys
import numpy
import csv
import os
# import pandas as pd
from math import exp, log
import matplotlib.pyplot
from PIL import Image, ImageTk
from graphviz import Graph
from statistics import mean

from numpy.random import randint
from numpy.random import rand

from typing import List, Set, Dict, Tuple, Optional


def getFits(dir_path: str):
    with open(dir_path, "r",  errors="ignore") as f:
        final_line = f.readlines()[-2]
        final_line = final_line.split(',')
        return final_line[-1]


def main():
    stuff = [[[] for _ in range(4)] for _ in range(2)]
    stuff_names = [[[] for _ in range(4)] for _ in range(2)]
    removal = 0
    strictness = 0
    for dirpath, dirnames, files in os.walk('.'):
        for file_name in files:
            if file_name.__contains__("gen3"):
                if file_name.endswith(".csv"):
                    direc = os.path.join(dirpath, file_name)
                    if file_name.__contains__("rem50"):
                        removal = 0
                    elif file_name.__contains__("rem70"):
                        removal = 1
                    if file_name.__contains__("strict"):
                        strictness = 0
                    elif file_name.__contains__("medium"):
                        strictness = 1
                    elif file_name.__contains__("lax"):
                        strictness = 2
                    elif file_name.__contains__("leery"):
                        strictness = 3
                    fit = getFits(direc)
                    stuff[removal][strictness].append(fit)
                    stuff_names[removal][strictness].append(file_name)
    
    best_strict50 = numpy.argmax(stuff[0][0])
    
    best_strict70 = numpy.argmax(stuff[1][0])

                    
                    




    return


if __name__ == "__main__":
    main()