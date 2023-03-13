import matplotlib.pyplot as plt

if __name__ == '__main__':

    labels=[r'$x=2$', r'$x=3$', r'$x=4$']
    titles=[r'$f_1(x)=|x|$', r'$f_2(x)=x^2$', r'$f_3(x)=2x^2$', r'$f_4(x)=x^2+20$']
    probabilities = [[2/7, 3/7, 4/7],
                     [4/29, 9/29, 16/29],
                     [4/29, 9/29, 16/29],
                     [24/89, 29/89, 36/89]]

    fig, ax = plt.subplots(1, 4, figsize=(10,6))
    for i,p, in enumerate(probabilities):
        ax[i].pie(p, labels=labels)
        ax[i].set_title(titles[i])

    plt.savefig('fitness_function_plot')
    plt.show()




