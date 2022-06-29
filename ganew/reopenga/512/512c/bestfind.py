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
        final_line = final_line.strip()
        final_line = final_line.split(',')
        return float(final_line[-1])


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
    
    with open("bestofbest.txt", mode='w')  as f:
        best_strict50 = numpy.argmax(stuff[0][0])
        best_strict50 = stuff_names[0][0][best_strict50]
        print(best_strict50, file=f)
        best_strict70 = numpy.argmax(stuff[1][0])
        best_strict70 = stuff_names[1][0][best_strict70]
        print(best_strict70, file=f)
        best_medium50 = numpy.argmax(stuff[0][1])
        best_medium50 = stuff_names[0][1][best_medium50]
        print(best_medium50, file=f)
        best_medium70 = numpy.argmax(stuff[1][1])
        best_medium70 = stuff_names[1][1][best_medium70]
        print(best_medium70, file=f)
        best_lax50 = numpy.argmax(stuff[0][2])
        best_lax50 = stuff_names[0][2][best_lax50]
        print(best_lax50, file=f)
        best_lax70 = numpy.argmax(stuff[1][2])
        best_lax70 = stuff_names[1][2][best_lax70]
        print(best_lax70, file=f)
        best_leery50 = numpy.argmax(stuff[0][3])
        best_leery50 = stuff_names[0][3][best_leery50]
        print(best_leery50, file=f)
        best_leery70 = numpy.argmax(stuff[1][3])
        best_leery70 = stuff_names[1][3][best_leery70]
        print(best_leery70, file=f)

                    
                    




    return


if __name__ == "__main__":
    main()