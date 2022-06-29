a256 = "10 256N_graph10.dat"
b256 = "19 256N_graph19.dat"
c256 = "22 256N_graph22.dat"
a512 = "3 512N_graph3.dat"
b512 = "9 512N_graph9.dat"
c512 = "18 512N_graph18.dat"
a768 = "5 768N_graph5.dat"
b768 = "21 768N_graph21.dat"
c768 = "24 768N_graph24.dat"
a1024 = "18 1024N_graph18.dat"
b1024 = "24 1024N_graph24.dat"
c1024 = "26 1024N_graph26.dat"

strict = " 0.025 0.01"
medium = " 0.05 0.02"
lax = " 0.075 0.03"
leery = " 0.1 0.04"

rem50 = " 0.5"
rem70 = " 0.7"

pop101 = " 101"
pop201 = " 201"

graph1 = " 256N_graph10"
# graph2 = " 256N_graph19"
# graph3 = " 256N_graph22"
graph4 = " 512N_graph3"
# graph5 = " 512N_graph9"
# graph6 = " 512N_graph18"
graph7 = " 768N_graph5"
# graph8 = " 768N_graph21"
# graph9 = " 768N_graph24"
graph10 = " 1024N_graph18"
# graph11 = " 1024N_graph24"
# graph12 = " 1024N_graph26"

gen1 = " 0.5 0.5"
gen2 = " 0.7 0.3"
gen3 = " 0.9 0.1"

genetics = {"gen1": " 0.5 0.5"}
graphs = [graph1, graph4, graph7, graph10]
expers = [a256, a512, a768, a1024]
pops = [pop101, pop201]
rems = {"rem50_": " 0.5", "rem70_": " 0.7"}
toughness = {"strict": " 0.025 0.01", "medium": " 0.05 0.02",
             "lax": " 0.075 0.03", "leery": " 0.1 0.04"}

with open("params.dat", 'w') as f:
    seed = 21*13*13*13*13
    for a in range(len(graphs)):
        for b in toughness:
            for c in rems:
                for d in genetics:
                    path = graphs[a] + b + d + c
                    print("python3 garandom.py " + expers[a] + toughness[b] + rems[c] + genetics[d] + " " + str(201) + path + " " + str(seed), file=f)



