import numpy as np
import random
import math
import matplotlib.pyplot as plt
from itertools import combinations
import copy
from tqdm import tqdm
import time
import tsplib95

# calculate distance (fitness) of permutation
def calculate_fitness(cities):
    distance = 0
    for i in range(len(cities)):
        distance += math.dist(cities[i], cities[i-1])
    return distance

# calculate the best fitness of entire generation
def best_fitness(individuals):
    fitnesses = []
    for individual in individuals:
        fitnesses.append(calculate_fitness(individual))
    idx = fitnesses.index(min(fitnesses))
    indi = individuals[idx]
    return (min(fitnesses), indi)

def mean_fitness(individuals):
    fitnesses = []
    for individual in individuals:
        fitnesses.append(calculate_fitness(individual))
    return np.mean(fitnesses)

# generate parents in generation using tournaments of size k
def tournament_selection(generation, k):
    parents = []
    n_battles = len(generation)//k
    for i in range(n_battles):
        best_fit = math.inf
        best_permutation = []
        for j in range(k):
            permutation = generation.pop(random.randint(0, len(generation)-1))
            fitness = calculate_fitness(permutation)
            if fitness < best_fit:
                best_fit = fitness
                best_permutation = permutation
        parents.append(best_permutation)
    return parents

# make crossovers of parents
def order_crossover(individuals, p_c):
    crossed = []
    # generate possible parent pairs
    all_combis = list(combinations(individuals, 2))
    # select parents
    selected_combis = random.sample(all_combis, len(individuals))

    # generate crossed over offspring
    for combi in selected_combis:
        indi1 = combi[0]
        indi2 = combi[1]
        if random.random() <= p_c:
            idx1, idx2 = np.sort(random.sample(range(0, len(indi1)+1), 2))
            slice1 = indi1[idx1:idx2]
            slice2 = indi2[idx1:idx2]

            missing1 = []
            for city in indi2:
                if city not in slice1:
                    missing1.append(city)

            missing2 = []
            for city in indi1:
                if city not in slice2:
                    missing2.append(city)

            newindi1 = []
            for i in range(len(indi1)):
                if i < idx1 or i >= idx2:
                    newindi1.append(missing1.pop(0))
                else:
                    newindi1.append(slice1[i-idx1])

            newindi2 = []
            for i in range(len(indi2)):
                if i < idx1 or i >= idx2:
                    newindi2.append(missing2.pop(0))
                else:
                    newindi2.append(slice2[i-idx1])

            crossed.append(newindi1)
            crossed.append(newindi2)

        else:
            crossed.append(indi1)
            crossed.append(indi2)

    return crossed

# generate mutations of parents
def mutations(individuals, mu):
    mutated = []
    for indi in individuals:
        if random.random() < mu:
            idxs = random.sample(range(0, len(indi)), 2)
            item1 = indi[idxs[0]]
            item2 = indi[idxs[1]]
            indi[idxs[0]] = item2
            indi[idxs[1]] = item1
            mutated.append(indi)
        else:
            mutated.append(indi)
    return mutated

# swap 2 cities using the 2-opt algorithm
def swap(cities, i, j):
    newcities = []
    newcities += cities[0:i+1]
    middle = list(reversed(cities[i+1:j+1]))
    newcities += middle
    newcities += cities[j+1:len(cities)]
    return newcities

# do a local search on a generation of cities, return the new and improved generation
def local_search(individuals):
    new_gen = []
    for indi in individuals:
        best_indi = indi
        best_delta = math.inf
        for i in range(len(indi)):
            for j in range(i+1, len(indi)):
                lengthDelta = - math.dist(indi[i], indi[(i+1)% len(indi)]) - math.dist(indi[j], indi[(j+1)% len(indi)]) + math.dist(indi[(i+1)% len(indi)], indi[(j+1)% len(indi)]) + math.dist(indi[i], indi[j])
                if lengthDelta < best_delta:
                    new_indi = swap(copy.deepcopy(indi), i, j)
                    best_delta = lengthDelta
                    best_indi = new_indi
        new_gen.append(best_indi)
    return new_gen      

def plot_fitness(best_fitnesses, mean_fitnesses):
    # plot fitness over time
    fig = plt.figure()
    plt.plot(mean_fitnesses)
    plt.plot(best_fitnesses)
    plt.legend(['mean fitness', 'best fitness'])
    plt.xlabel("epochs")
    plt.ylabel("shortest distance/fitness")
    plt.savefig('ex3pics/fitness.png')

def plot_route(cities):
    fig = plt.figure()
    plt.scatter(*zip(*cities), color='green')
    for i in range(len(cities)):
        plt.plot((cities[i][0], cities[i-1][0]), (cities[i][1], cities[i-1][1]), color='red')
    plt.title("Fastest route found")
    plt.savefig('ex3pics/bestroute.png')


def solve_tsp(p_c, mu, memetic):

    # maximum city coordinates
    max_x = 50.0
    max_y = 50.0
    # number of cities in the TSP
    n_cities = 50

    # number of individuals in every generation
    generation_size = 20
    generation = []

    # generate cities with random float x and y coordinate
    cities = []
    # for i in range(n_cities):
    #     cities.append((round(random.uniform(0, max_x), 4), round(random.uniform(0, max_y), 4)))
    
    # read cities from file
    # with open('file-tsp.txt') as f:
    #     for line in f: # 
    #         cities.append([float(x) for x in line.split()])
    # print(cities)

    # load tsp file
    cities = list(tsplib95.load('berlin52.tsp').as_name_dict()['node_coords'].values())

    # create random permutations of order
    for i in range(generation_size):
        generation.append(cities)
        cities = random.sample(cities, len(cities)) 

    # start evolving
    best_fitnesses = []
    mean_fitnesses = []
    epochs = 1500
    for i in tqdm(range(epochs)):
        # calculate best and avg fitness
        best_fitnesses.append(best_fitness(generation)[0])
        mean_fitnesses.append(mean_fitness(generation))

        # generate parents
        parents = tournament_selection(generation, k=2)

        # recombine and mutate
        crossover = order_crossover(parents, p_c)
        generation = mutations(crossover, mu)

        if memetic:
            generation = local_search(generation)

    plot_fitness(best_fitnesses, mean_fitnesses)
    plot_route(best_fitness(generation)[1])
    print("best fitness found: " + str(min(best_fitnesses)))

if __name__ == "__main__":
    for i in range(10):
        random.seed(i)
        solve_tsp(p_c=1, mu=0.01, memetic=True)