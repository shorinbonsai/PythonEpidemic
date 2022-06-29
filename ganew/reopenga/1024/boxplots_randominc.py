from lib2to3.pgen2.tokenize import generate_tokens
import sys
import os
import csv
import math
import re
import string
from operator import itemgetter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="ticks", color_codes=True)


def getFits(dir_path: str):
    with open(dir_path, "r",  errors="ignore") as f:
        final_line = f.readlines()[-2]
        final_line = final_line.split(',')
        return final_line[-1]

def getRandom(dir_path: str) -> list[int]:
    result = []
    with open(dir_path, "r",  errors="ignore") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            result.append(float(line))
    return result


def main():
    stuff = [[[] for _ in range(4)] for _ in range(4)]
    stuff2 = [[[] for _ in range(4)] for _ in range(4)]
    for dirpath, dirnames, files in os.walk('.'):
        for file_name in files:
            if file_name.endswith(".csv"):
                direc = os.path.join(dirpath, file_name)
                exper = []
                exper = dirpath.split('/')
                if file_name.__contains__("random"):
                    best = getRandom(direc)
                    genetics = 3
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
                    if removal == 0:
                        stuff[genetics][strictness] = best
                    elif removal == 1:
                        stuff2[genetics][strictness] = best
                    
                    
                else: 
                    best = getFits(direc)
                    genetics = 6
                    strictness = 12
                    removal = 5
                    if file_name.__contains__("gen1"):
                        genetics = 0
                    elif file_name.__contains__("gen2"):
                        genetics = 1
                    elif file_name.__contains__("gen3"):
                        genetics = 2
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
                    if removal == 0:
                        stuff[genetics][strictness].append(best)
                    elif removal == 1:
                        stuff2[genetics][strictness].append(best)


    df = pd.DataFrame()
    gene1 = stuff[0]
    gene2 = stuff[1]
    gene3 = stuff[2]
    gene4 = stuff[3]
    experraw = ["Strict", "Medium", "Lax", "Skeptical"]
    df = pd.DataFrame(
        {'Parameters': experraw, 'Random': gene4, '50/50': gene1, '70/30': gene2, '90/10': gene3})
    df = df[['Parameters','Random', '50/50', '70/30', '90/10' ]]
    dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
                 'Random', '50/50', '70/30', '90/10' ], var_name='Experiments')
    result = dd.explode('value')
    result['value'] = result['value'].astype('float')
    ax = sns.boxplot(x='Parameters', y='value', data=result,
                     hue='Experiments')
    plt.legend([],[], frameon=False)
    ax.set(xlabel="Parameters", ylabel="Total Infected")
    ax.set_title("50% Removal")
    plt.show()


    df = pd.DataFrame()
    gene1 = stuff2[0]
    gene2 = stuff2[1]
    gene3 = stuff2[2]
    gene4  =stuff2[3]
    experraw = ["Strict", "Medium", "Lax", "Skeptical"]
    df = pd.DataFrame(
        {'Parameters': experraw, 'Random': gene4, '50/50': gene1, '70/30': gene2, '90/10': gene3})
    df = df[['Parameters','Random', '50/50', '70/30', '90/10' ]]
    dd = pd.melt(df, id_vars=['Parameters'], value_vars=[
                 'Random', '50/50', '70/30', '90/10'], var_name='Experiments')
    result = dd.explode('value')
    result['value'] = result['value'].astype('float')
    ax = sns.boxplot(x='Parameters', y='value', data=result,
                     hue='Experiments')
    plt.legend([],[], frameon=False)
    ax.set(xlabel="Parameters", ylabel="Total Infected")
    ax.set_title("70% Removal")
    plt.show()


    return


if '__main__' == __name__:
    main()
