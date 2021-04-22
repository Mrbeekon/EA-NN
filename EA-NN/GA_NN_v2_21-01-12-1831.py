#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:04:19 2020

@author: sam

"""

import random
import math
import numpy
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import KFold
from datetime import datetime
from copy import deepcopy


# NN constants
# each number represents the nodes in the hidden layer then the last is the output nodes
# e.g. [3, 2, 1] 2 hidden layers, the first has 3 nodes the second has 2 nodes and there is 1 output node
NN_STRUCTURE = [5, 1]

# GA constants
POP_SIZE = 30  # Population size
MAX_GEN = 500  # Generations
MUT_RATE_START = 0.01  # Mutation rate
MUT_STEP_START = 1.0  # Mutation step
CROSS_RATE = 0.75  # Crossover rate
UPPER = 1.0  # Upper gene limit
LOWER = -1.0  # Lower gene limit

K = 10


class individual:
    gene = []
    fitness = 0


def get_data_set(filename):
    txt = open(filename, "r")
    lines = txt.readlines()
    txt.close()

    out_list = []
    for line in lines:
        temp_line = []
        line = line.split(" ")
        line[-1] = line[-1].rsplit("\n")[0]
        for el in line:
            temp_line.append(float(el))
        out_list.append(temp_line)
    return out_list


def initialise_pop(gene_len):
    # Set initial values
    pop = []
    for i in range(POP_SIZE):
        temp_gene = []
        for j in range(gene_len):
            temp_gene.append(round(random.uniform(UPPER, LOWER), 2))
        new_ind = individual()
        new_ind.gene = temp_gene
        pop.append(new_ind)
    return pop


def select(pop):
    off = []
    for i in range(len(pop)):
        parent1 = pop[i]
        off1 = parent1

        parent2 = pop[random.randint(0, POP_SIZE - 1)]
        off2 = parent2

        parent3 = pop[random.randint(0, POP_SIZE - 1)]
        off3 = parent3

        if parent1.fitness < parent2.fitness:
            if parent1.fitness < parent3.fitness:
                off.append(off1)
            else:
                off.append(off3)
        else:
            if parent2.fitness < parent3.fitness:
                off.append(off2)
            else:
                off.append(off3)
    return off


# tail swap
def crossover(ind1, ind2):
    # print('Before crossover = ', ind1, ind2)
    slicey = random.randint(0, len(ind1)-1)

    # get head and tail
    ind1_head = ind1[:slicey]
    ind1_tail = ind1[slicey:]

    ind2_head = ind2[:slicey]
    ind2_tail = ind2[slicey:]

    ind1 = ind1_head + ind2_tail
    ind2 = ind2_head + ind1_tail
    return ind1, ind2


def mutate(ind, j, step):
    if random.randint(0, 100) % 2 != 0:
        alter = random.uniform(0, step)
        ind.gene[j] += alter
    else:
        alter = random.uniform(0, step)
        ind.gene[j] -= alter

    ind.gene[j] = round(ind.gene[j], 2)
    return ind


def plot_graph(graph, data_name):
    plt.plot(graph[0], graph[1], label='avg. fitness (train)')
    plt.plot(graph[0], graph[2], label='best fitness (train)')
    plt.plot(graph[0], graph[3], label='best fitness (test)')
    mini = min(graph[3])
    plt.hlines(y=mini, xmin=0, xmax=MAX_GEN,
               label='min ' + "{:.2f}".format(mini),
               color='grey')
    y_max = 100
    plt.axis([0, MAX_GEN, 0, y_max])
    plt.xlabel('Generations')
    plt.ylabel('Error (%)')
    plt.legend()

    title = 'Pop. size = ' + str(POP_SIZE) + ' Max gen. = ' + \
            str(MAX_GEN) + '\n Mut. rate start = ' + str(MUT_RATE_START) + ' Mut. step start= ' + str(MUT_STEP_START) + \
            ' Cross rate = ' + str(CROSS_RATE) + '\n Gene Upper = ' + str(UPPER) + ' Gene Lower = ' + \
            str(LOWER) + '\n NN Structure = ' + str(NN_STRUCTURE) + ' K = ' + str(K)

    plt.rcParams["axes.titlesize"] = 8
    plt.title(title)

    plt.savefig('NN_plots/GA_NN_v2_' + data_name.split("/")[-1].split(".")[0] + str(datetime.now()) + '.png')
    plt.show()


def sigmoid(x):
    return 1.0/(1.0 + numpy.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    t = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
    return t


def do_nn_layer(input_node_values, weights, bias):
    len_input = len(input_node_values)

    # print('------')
    # reshape for dot product
    input_array = numpy.reshape(numpy.array(input_node_values), (1, len_input))
    weight_array = numpy.reshape(numpy.array(weights), (len_input, int(len(weights)/len_input)))
    layer = numpy.dot(input_array, weight_array)[0] # multiplying 2 arrays and sum
    # print('dot =                    ', layer)

    # add bias
    layer += bias
    # print('dot + bias =             ', layer)

    # apply activation
    layer_node_values = []

    for el in layer:
        layer_node_values.append(sigmoid(el))
    layer = layer_node_values
    # print('activation(dot + bias) = ', layer)
    return layer


def ga_start(training_data, gene_structure, test_data):
    gen = 0
    graph = [[], [], [], []]

    training_data.tolist()
    test_data.tolist()

    mut_rate = MUT_RATE_START
    mut_step = MUT_STEP_START

    # initialise population
    population = initialise_pop(gene_len)

    while gen <= MAX_GEN:
        # [print(ind.gene) for ind in population]

        # calculate fitness
        tot_fitness = 0
        for ind in population:
            # slice up gene
            sliced_gene = []
            start_slice = 0
            end_slice = 0
            for weight_set in gene_structure:
                end_slice += weight_set
                sliced_gene.append(ind.gene[start_slice:end_slice])
                start_slice += weight_set

            # fitness will be total error of gene
            error = []
            # starting nn
            for entry in training_data:
                input_data = entry[:-1]
                for i in range(len(NN_STRUCTURE)):
                    input_data = do_nn_layer(input_data, sliced_gene[i], sliced_gene[i+len(NN_STRUCTURE)])
                nn_output = round(input_data[0], 0)

                # sanity checking
                if nn_output > 1:
                    nn_output = 1
                if nn_output < 0:
                    nn_output = 0

                # check if nn was correct or not
                if entry[-1] == nn_output:
                    error.append(0)
                else:
                    error.append(1)
            ind.fitness = (sum(error)/len(error))*100
            tot_fitness += ind.fitness
        avg_fitness = tot_fitness / POP_SIZE

        # sort population for getting best/worst
        population.sort(key=lambda x: x.fitness)
        best_ind = copy.deepcopy(population[0])

        # graph data
        graph[0].append(gen)  # x
        graph[1].append(avg_fitness)  # y
        graph[2].append(best_ind.fitness)  # y2
        graph[3].append(do_test(test_data, copy.deepcopy(best_ind), gene_structure))  # y3

        # console prints
        print('\n ----- GEN ', gen, ' ----- Best individual  = ', best_ind.fitness)

        # select
        offspring = select(population)

        # crossover
        for i in range(POP_SIZE):
            if random.uniform(0, 1) < CROSS_RATE:
                random_index = random.randint(0, POP_SIZE-1)
                cross = crossover(offspring[i].gene, offspring[random_index].gene)
                offspring[i].gene = cross[0]
                offspring[random_index].gene = cross[1]

        # mutate
        mut_rate = mut_rate + ((1-MUT_RATE_START)/MAX_GEN)
        mut_step = mut_step - (MUT_STEP_START/MAX_GEN)
        for ind in offspring:
            for j in range(len(ind.gene)):
                if random.uniform(0, 1) < mut_rate:
                    ind = mutate(ind, j, mut_step)

        # sort for getting worst
        offspring[-1] = best_ind

        # new population is current offspring
        population = offspring

        gen += 1

    # show performance
    return graph, best_ind


def do_test(test_data, candidate, gene_structure):
    # fitness will be total error of gene
    # slice up gene
    sliced_gene = []
    start_slice = 0
    end_slice = 0
    for weight_set in gene_structure:
        end_slice += weight_set
        sliced_gene.append(candidate.gene[start_slice:end_slice])
        start_slice += weight_set

    # fitness will be total error of gene
    error = []
    # starting nn
    for entry in test_data:
        input_data = entry[:-1]
        for i in range(len(NN_STRUCTURE)):
            input_data = do_nn_layer(input_data, sliced_gene[i], sliced_gene[i + len(NN_STRUCTURE)])
        nn_output = round(input_data[0], 0)

        # check if nn was correct or not
        if entry[-1] == nn_output:
            error.append(0)
        else:
            error.append(1)

    candidate.fitness = (sum(error) / len(error)) * 100
    return candidate.fitness


if __name__ == "__main__":
    data_set_path = "data/data2.txt"
    data = get_data_set(data_set_path)
    # split data into test and train
    len_data = len(data)

    # setup gene structure for conversion to nn
    gene_shape = [(len(data[0]) - 1) * NN_STRUCTURE[0]]
    for i in range(len(NN_STRUCTURE)):
        try:
            gene_shape.append(NN_STRUCTURE[i] * NN_STRUCTURE[i + 1])
        except IndexError:
            break

    # bias weights
    for i in NN_STRUCTURE:
        gene_shape.append(i)

    # display nn structure
    print('==== NN Structure ====')
    print('NN = ', NN_STRUCTURE)
    print('WEIGHT LINES =', gene_shape)
    print('LAYER WEIGHTS = ', gene_shape[:-(len(NN_STRUCTURE))], ' BIAS WEIGHTS ',
          gene_shape[-(len(NN_STRUCTURE)):], '\n')

    print(len(data[0]) - 1, 'INPUT NODES')
    print(gene_shape[0], 'WEIGHTS')
    for i in range(len(NN_STRUCTURE) - 1):
        print(NN_STRUCTURE[i], 'HIDDEN LAYER', i, 'NODES')
        print(gene_shape[i + 1], 'WEIGHTS')
    print(NN_STRUCTURE[-1], 'OUTPUT NODES')
    gene_len = sum(gene_shape)
    print('\nGENE LENGTH = ', gene_len)
    print('======================')

    # Kfold
    kfold = KFold(K, True, 1)
    data = numpy.array(data)
    # start training
    list_of_graphs = []
    for train, test in kfold.split(data):
        list_of_graphs.append(numpy.array(ga_start(data[train], gene_shape, data[test])[0]))

    for i in range(len(list_of_graphs)):
        try:
            list_of_graphs[i + 1] = numpy.add(list_of_graphs[i], list_of_graphs[i + 1])
        except IndexError:
            # to show done
            print('beep')
            break

    plot_graph(list_of_graphs[-1] / len(list_of_graphs), data_set_path)
