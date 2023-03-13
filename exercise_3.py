import numpy as np
import random
import math
import matplotlib.pyplot as plt
from itertools import combinations
import copy
from tqdm import tqdm
import time

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
    return min(fitnesses)

def mean_fitness(individuals):
    fitnesses = []
    for individual in individuals:
        fitnesses.append(calculate_fitness(individual))
    return np.mean(fitnesses)

# generate parents in generation using tournaments of size k
def tournament_selection(generation, k):
    # print("bestgen:")
    # print(str(mean_fitness(generation)))
    parents = []
    n_battles = len(generation)//k
    for i in range(n_battles):
        best_fit = math.inf
        best_permutation = []
        for j in range(k):
            # print(len(generation))
            permutation = generation.pop(random.randint(0, len(generation)-1))
            fitness = calculate_fitness(permutation)
            if fitness < best_fit:
                best_fit = fitness
                best_permutation = permutation
        parents.append(best_permutation)
    # print("bestparents:")
    # print(str(mean_fitness(parents)))
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
        # print(len(indi1))
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

def plot_fitness(best_fitnesses, mean_fitnesses):
    # plot fitness over time
    plt.plot(mean_fitnesses)
    plt.plot(best_fitnesses)
    plt.legend(['mean fitness', 'best fitness'])
    plt.xlabel("epochs")
    plt.ylabel("shortest distance/fitness")
    plt.savefig('fitness.png')

def plot_route(cities):
    plt.scatter(*zip(*cities), color='green')
    for i in range(len(cities)):
        plt.plot((cities[i][0], cities[i-1][0]), (cities[i][1], cities[i-1][1]), color='red')
    plt.savefig('bestroute.png')


def solve_tsp(p_c, mu):
    random.seed(1)

    # maximum city coordinates
    max_x = 50.0
    max_y = 50.0
    # number of cities in the TSP
    n_cities = 50

    # number of individuals in every generation
    generation_size = 500
    generation = []

    # generate cities with random float x and y coordinate
    cities = []
    for i in range(n_cities):
        cities.append((round(random.uniform(0, max_x), 4), round(random.uniform(0, max_y), 4)))

    # create random permutations of order
    for i in range(generation_size):
        generation.append(cities)
        cities = random.sample(cities, len(cities)) 

    # start evolving
    best_fitnesses = []
    mean_fitnesses = []
    epochs = 1500
    for i in tqdm(range(epochs)):
        best_fitnesses.append(best_fitness(generation))
        mean_fitnesses.append(mean_fitness(generation))
        parents = tournament_selection(generation, k=2)
        crossover = order_crossover(parents, p_c)
        generation = mutations(crossover, mu)

    plot_fitness(best_fitnesses, mean_fitnesses)
    plot_route(generation[0])

if __name__ == "__main__":
    solve_tsp(p_c=1, mu=0.01)