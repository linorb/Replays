import numpy as np
import matplotlib.pyplot as plt

def main():
    ##  Plot statistics of the linear track

    MOUSE = [3, 6, 6, 4, 3, 0, 4, 4, 1, 1]
    CAGE = [40, 40, 38, 38, 38, 38, 6, 7, 11, 13]
    mouse_color = [(1, 0, 0), #red
                   (0, 1, 0), #green
                   (0, 0, 1), #blue
                   (1, 1, 0), #
                   (0, 1, 1), #cyan
                   (1, 0, 1), #purple
                   (1, 0.5, 0),
                   (0.5, 0, 0.5),
                   (0, 0.5, 0.5),
                   (0.5, 0.5, 0)]

    # Load the data
    filename = 'linear_edge_statistics.npz'
    npzfile = np.load(filename)
    p_value = npzfile['p_value'].all()
    t_value = npzfile['t_value'].all()
    cohen_d = npzfile['cohen_d'].all()

    # Plot Cohen's D:
    f = plt.figure()
    mouse_plot = []
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' %(CAGE[i], MOUSE[i])
        a, = plt.plot(np.array(cohen_d[mouse_name]['d_before']),
             np.array(cohen_d[mouse_name]['d_after']),
             markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                 markersize=8, label=mouse_name)
        mouse_plot.append(a)

    plt.legend(handles=mouse_plot)
    plt.plot(np.zeros(21), np.arange(-1, 1.1, 0.1), 'k')
    plt.plot(np.arange(-1, 1.1, 0.1), np.zeros(21), 'k')
    plt.grid()
    plt.ylabel(
        'Effect size of: p(active after run|active in run) - '
        'p(active after run|not active in run)', fontsize=17)
    plt.xlabel(
        'Effect size of: p(active before run|active in run) -'
        ' p(active before run|not active in run)', fontsize=17)
    plt.title('Linear Track', fontsize=25)
    plt.xticks(np.arange(-1,1,0.1))
    plt.yticks(np.arange(-1, 1, 0.1))

    f.show()
    raw_input('press enter to quit')

if __name__ == '__main__':
    main()