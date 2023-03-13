import numpy as np
import matplotlib.pyplot as plt
import string
import random
from tqdm import tqdm

def mutate(member, sigma, mu):

    mutated = member
    for i in range(len(member)):
        sigma_copy = sigma.copy()
        sigma_copy.remove(member[i])
        if np.random.rand() <= mu:
            mutated[i] = random.sample(sigma_copy, k=1)[0]

    return mutated

def string_search_ga(target, pc, mu, N, K, g_max=100, verbose=False):

    # convert target to string
    target = [*target]
    # generate alphabet
    sigma = [*string.ascii_lowercase]+[' ']
    # define fitness function
    fitness = lambda member: np.sum(target==member)
    # create population
    population = np.array([np.array(random.choices(sigma, k=len(target))) for i in range(N)])
    # keep record of fittest member of each generation
    elite = population[np.argmax([fitness(member) for member in population])]
    f_elite = fitness(elite)
    fittest = [elite]
    g = 0

    # while not target string yet or below max generations
    while f_elite < len(target) and g < g_max:

        # create new population
        new_population = []
        for i in range(0, N-2, 2):

            # find new parents
            parents = []
            for j in range(2):

                # get indices of tournament participants
                tournament_idx = random.sample(range(len(population)), K)
                fs = [fitness(member) for member in population[tournament_idx]]
                winner = np.argmax(fs)
                parents.append(population[tournament_idx[winner]])

            # apply crossover between parents with probability pc and generate offspring
            if np.random.rand() <= pc:
                co_location = np.random.randint(0, len(target))
                offspring_1 = np.concatenate((parents[0][0:co_location],parents[1][co_location:]))
                offspring_2 = np.concatenate((parents[1][0:co_location],parents[0][co_location:]))

            else:
                offspring_1 = parents[0]
                offspring_2 = parents[1]

            # apply mutation to offspring
            new_population.append(mutate(offspring_1, sigma, mu))
            new_population.append(mutate(offspring_2, sigma, mu))

        population = np.array(new_population)
        # find fittest member of new population
        elite = new_population[np.argmax([fitness(member) for member in population])]
        fittest.append(elite)
        f_elite = fitness(elite)
        g += 1

    return fittest, g

def plot_t_finish(target, pc, mu, N, K):

    tf = []
    n = 20
    for i in range(n):
        fittest, n_gen = string_search_ga(target, pc, mu, N, K)
        tf.append(n_gen)

    plt.hist(tf)
    plt.axvline(np.mean(tf), color='red', label='average')
    plt.legend()
    plt.xlabel(r'$t_{finish}$')
    plt.ylabel('Frequency')
    plt.title(r'Finishing Times "hello world", $K=2$, $N=1000$, $\mu=0.1$, $p_c=1$, $G_{max}=100$')
    plt.show()

def plot_optimal_mu(target, pc, N, K):

    mus = np.linspace(0, 0.1, 50)
    tf_averages = []
    tf_stds = []
    n = 20

    for mu in tqdm(mus):
        tf = []
        for i in range(n):
            fittest, n_gen = string_search_ga(target, pc, mu, N, K)
            tf.append(n_gen)
        tf_averages.append(np.mean(tf))
        tf_stds.append(np.std(tf))

    plt.plot(mus, tf_averages, label=r'average $t_{finish}$')
    plt.plot(mus, tf_stds, label='std $t_{finish}$')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$t_{finish}$')
    plt.title(r'$\mu$ vs. $t_{finish}$, K=2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # plot_t_finish(target='hello world', pc=1, mu=0.1, N=1000, K=2)
    plot_optimal_mu(target='hello world', pc=1, N=1000, K=2)
