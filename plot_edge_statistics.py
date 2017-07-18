import numpy as np
import matplotlib.pyplot as plt

def main():
    ##  Plot statistics of the linear track

    ######### For linear track #########:

    ### Load the data ###
    filename = 'linear_edge_statistics.npz'
    npzfile = np.load(filename)
    p_value = npzfile['p_value'].all()
    t_value = npzfile['t_value'].all()
    cohen_d = npzfile['cohen_d'].all()

    MOUSE = [3, 6, 6, 4, 3, 0, 4, 4, 1, 1]
    CAGE = [40, 40, 38, 38, 38, 38, 6, 7, 11, 13]
    mouse_color = [(1, 0, 0),  # red
                   (0, 1, 0),  # green
                   (0, 0, 1),  # blue
                   (1, 1, 0),  #
                   (0, 1, 1),  # cyan
                   (1, 0, 1),  # purple
                   (1, 0.5, 0),
                   (0.5, 0, 0.5),
                   (0, 0.5, 0.5),
                   (0.5, 0.5, 0)]

    ### Plot Cohen's D ###:
    f0 = plt.figure()
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
    plt.xticks(np.arange(-1, 1, 0.1), fontsize=15)
    plt.yticks(np.arange(-1, 1, 0.1), fontsize=15)

    # f0.show()

    ### P value Vs t value ###
    f, axx = plt.subplots(1, 2, sharex=True, sharey=True)
    mouse_legend = []
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        axx[0].plot(np.array(t_value[mouse_name]['t_before']),
                  np.array(p_value[mouse_name]['p_before']),
                  markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                  markersize=8)

        a, = axx[1].plot(np.array(t_value[mouse_name]['t_after']),
                         np.array(p_value[mouse_name]['p_after']),
                         markerfacecolor=mouse_color[i], marker='o',
                         linestyle='None', markersize=8, label=mouse_name)
        mouse_legend.append(a)

    plt.legend(handles=mouse_plot)
    axx[0].axhline(0, color='black')
    axx[0].axvline(0, color='black')
    axx[0].axhline(0.025, color='red')
    axx[0].grid()
    axx[0].set_xlabel('T statistic', fontsize=17)
    axx[0].set_ylabel('P value', fontsize=17)
    axx[0].set_title('p(active before run|active in run) - \n'
        'p(active before run|not active in run)')

    plt.xticks(np.arange(-10, 10, 1), fontsize=13)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
    axx[1].axhline(0, color='black')
    axx[1].axvline(0, color='black')
    axx[1].axhline(0.025, color='red')
    axx[1].set_xlabel('T statistic', fontsize=17)
    axx[1].set_ylabel('P value', fontsize=17)
    axx[1].set_title('p(active after run|active in run) - \n'
        'p(active after run|not active in run)')
    axx[1].set_ylim(-0.01, 1.1)
    axx[1].set_xlim(-10, 10)
    axx[1].grid(axis='both')

    f.suptitle('Linear Track', fontsize=25)

    f.show()

    ######### For L-shape track #########:

    ### Load the data ###
    filename = 'L-shape_edge_statistics.npz'
    npzfile = np.load(filename)
    p_value = npzfile['p_value'].all()
    t_value = npzfile['t_value'].all()
    cohen_d = npzfile['cohen_d'].all()

    MOUSE = [4, 4, 1, 1]
    CAGE = [6, 7, 11, 13]
    mouse_color = [(1, 0.5, 0),
                   (0.5, 0, 0.5),
                   (0, 0.5, 0.5),
                   (0.5, 0.5, 0)]

    ### Plot Cohen's D ###
    f0 = plt.figure()
    mouse_plot = []
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        a, = plt.plot(np.array(cohen_d[mouse_name]['d_before']),
                      np.array(cohen_d[mouse_name]['d_after']),
                      markerfacecolor=mouse_color[i], marker='o',
                      linestyle='None',
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
    plt.title('L-shape Track', fontsize=25)
    plt.xticks(np.arange(-1, 1, 0.1), fontsize=15)
    plt.yticks(np.arange(-1, 1, 0.1), fontsize=15)

    # f0.show()

    ###  P value Vs t value ###
    f, axx = plt.subplots(1, 2, sharex=True, sharey=True)
    mouse_legend = []
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        axx[0].plot(np.array(t_value[mouse_name]['t_before']),
                  np.array(p_value[mouse_name]['p_before']),
                  markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                  markersize=8)

        a, = axx[1].plot(np.array(t_value[mouse_name]['t_after']),
                         np.array(p_value[mouse_name]['p_after']),
                         markerfacecolor=mouse_color[i], marker='o',
                         linestyle='None', markersize=8, label=mouse_name)
        mouse_legend.append(a)

    plt.legend(handles=mouse_plot)
    axx[0].axhline(0, color='black')
    axx[0].axvline(0, color='black')
    axx[0].axhline(0.025, color='red')
    axx[0].grid()
    axx[0].set_xlabel('T statistic', fontsize=17)
    axx[0].set_ylabel('P value', fontsize=17)
    axx[0].set_title('p(active before run|active in run) - \n'
        'p(active before run|not active in run)')

    plt.xticks(np.arange(-10, 10, 1), fontsize=13)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=13)
    axx[1].axhline(0, color='black')
    axx[1].axvline(0, color='black')
    axx[1].axhline(0.025, color='red')
    axx[1].set_xlabel('T statistic', fontsize=17)
    axx[1].set_ylabel('P value', fontsize=17)
    axx[1].set_title('p(active after run|active in run) - \n'
        'p(active after run|not active in run)')
    axx[1].set_ylim(-0.01, 1.1)
    axx[1].set_xlim(-10, 10)
    axx[1].grid(axis='both')

    f.suptitle('L-shape Track', fontsize=25)

    f.show()

    raw_input('press enter to quit')

if __name__ == '__main__':
    main()