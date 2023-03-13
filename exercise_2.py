import numpy as np
import matplotlib.pyplot as plt

def simple_ga_counting_ones(l, n_gens, elitism=True):
    # mutation rate
    mu = 1/l
    # random bit sequence
    x = np.random.binomial(1, 0.5, l)
    # keep track of fitness at generations
    f = [np.sum(x)]
    f_max = f[0]

    for gen in range(1, n_gens):

        xm = x.copy()
        # determine which bits get inverted
        ps = np.random.rand(l)
        idx = np.where(ps <= mu)
        # invert bits
        xm[idx] = 1 - xm[idx]
        # compute fitness of xm
        if elitism:
            # keep fittest generation
            f_xm = np.sum(xm)
            if f_xm > f[gen-1]:
                x = xm
                f.append(f_xm)
            else:
                f.append(f[gen-1])
        # always update
        else:
            x = xm
            f.append(f_xm)

    return x, f

def plot_performance(l, n_gens):

    n_gens = 1500
    l = 100
    n_chains = 10

    fig, ax = plt.subplots(2, 5, figsize=(20, 6))
    fig.suptitle('Fitness versus Elapsed Generations With Elitism', y=0.94, fontsize=15)

    for c in range(n_chains):
        i = int(np.round(c/(n_chains-1)))
        j = c%5
        _, f = simple_ga_counting_ones(l, n_gens)
        ax[i,j].plot(range(n_gens), f)
        ax[1,j].set_xticks(np.arange(0,1501,500))
        ax[0,j].set_xticks([])
        ax[i,j].set_yticks(np.arange(0,101,50))
        ax[i,j].tick_params(labelsize=15)
        ax[1,j].set_xlabel(r'$n_{th}$-gen', fontsize=15, labelpad=0)
        ax[i,0].set_ylabel('fitness', fontsize=15, labelpad=-1)
    plt.show()

if __name__ == '__main__':
    plot_performance(l=100,n_gens=1500)




