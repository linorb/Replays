import numpy as np
import matplotlib.pyplot as plt

def main():
    ##  Plot statistics of the linear track
    f, axx = plt.subplots(2, 3)
    ######### For linear track #########:

    ### Load the data ###
    filename = r'results\linear_edge_statistics.npz'
    npzfile = np.load(filename)
    p_value = npzfile['p_value'].all()
    t_value = npzfile['t_value'].all()
    cohen_d = npzfile['cohen_d'].all()

    MOUSE = [3, 6, 6, 3, 0, 4, 4, 1, 1]
    CAGE = [40, 40, 38, 38, 38, 6, 7, 11, 13]
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
    mouse_plot = []
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' %(CAGE[i], MOUSE[i])
        a, = axx[0,2].plot(np.array(cohen_d[mouse_name]['d_before']),
             np.array(cohen_d[mouse_name]['d_after']),
             markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                 markersize=8, label=mouse_name)
        mouse_plot.append(a)

    # put the legend outside the axis. taken from:
    #  https://matplotlib.org/users/legend_guide.html
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
               handles=mouse_plot, fontsize=20)
    axx[0, 2].axhline(0, color='black')
    axx[0, 2].axvline(0, color='black')
    axx[0, 2].grid()
    axx[0, 2].set_ylabel('after run', fontsize=23)
    axx[0, 2].set_title('Effect size', fontsize=23)
    axx[0, 2].set_xlim(-1, 1)
    axx[0, 2].set_ylim(-1, 1)

    ### P value Vs t value ###

    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        axx[0, 0].plot(np.array(t_value[mouse_name]['t_before']),
                  np.array(p_value[mouse_name]['p_before']),
                  markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                  markersize=8)

        axx[0, 1].plot(np.array(t_value[mouse_name]['t_after']),
                     np.array(p_value[mouse_name]['p_after']),
                     markerfacecolor=mouse_color[i], marker='o',
                     linestyle='None', markersize=8, label=mouse_name)

    axx[0, 0].axhline(0, color='black')
    axx[0, 0].axvline(0, color='black')
    axx[0, 0].axhline(0.023, color='red')
    axx[0, 0].grid()
    axx[0, 0].set_ylabel('Environment A \n\nP value', fontsize=23)
    axx[0, 0].set_title('p(active before run|active in run) - \n'
        'p(active before run|not active in run)', fontsize=23)
    axx[0, 0].set_xlim(-10, 10)
    axx[0, 0].set_ylim(-0.03, 1.1)

    axx[0, 1].axhline(0, color='black')
    axx[0, 1].axvline(0, color='black')
    axx[0, 1].axhline(0.023, color='red')
    axx[0, 1].set_ylabel('P value', fontsize=23)
    axx[0, 1].set_title('p(active after run|active in run) - \n'
        'p(active after run|not active in run)', fontsize=23)
    axx[0, 1].set_ylim(-0.03, 1.1)
    axx[0, 1].set_xlim(-10, 10)
    axx[0, 1].grid(axis='both')


    ######### For L-shape track #########:

    ### Load the data ###
    filename = r'results\Lshape_edge_statistics.npz'
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
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        axx[1, 2].plot(np.array(cohen_d[mouse_name]['d_before']),
                      np.array(cohen_d[mouse_name]['d_after']),
                      markerfacecolor=mouse_color[i], marker='o',
                      linestyle='None', markersize=8)

    axx[1, 2].axhline(0, color='black')
    axx[1, 2].axvline(0, color='black')
    axx[1, 2].grid()
    axx[1, 2].set_ylabel(
        'after run', fontsize=23)
    axx[1, 2].set_xlabel(
        'before run', fontsize=23)
    axx[1, 2].set_xlim(-1, 1)
    axx[1, 2].set_ylim(-1, 1)


    ###  P value Vs t value ###
    for i in range(len(MOUSE)):
        mouse_name = 'c%dm%d' % (CAGE[i], MOUSE[i])
        axx[1, 0].plot(np.array(t_value[mouse_name]['t_before']),
                  np.array(p_value[mouse_name]['p_before']),
                  markerfacecolor=mouse_color[i], marker='o', linestyle='None',
                  markersize=8)

        axx[1, 1].plot(np.array(t_value[mouse_name]['t_after']),
                         np.array(p_value[mouse_name]['p_after']),
                         markerfacecolor=mouse_color[i], marker='o',
                         linestyle='None', markersize=8, label=mouse_name)

    axx[1, 0].axhline(0, color='black')
    axx[1, 0].axvline(0, color='black')
    axx[1, 0].axhline(0.023, color='red')
    axx[1, 0].grid()
    axx[1, 0].set_xlabel('T statistic', fontsize=23)
    axx[1, 0].set_ylabel('Environment B \n\nP value', fontsize=23)
    # axx[1, 0].set_aspect('equal', 'datalim')
    axx[1, 0].set_xlim(-10, 10)
    axx[1, 0].set_ylim(-0.03, 1.1)

    axx[1,  1].axhline(0, color='black')
    axx[1,  1].axvline(0, color='black')
    axx[1,  1].axhline(0.023, color='red')
    axx[1,  1].set_xlabel('T statistic', fontsize=23)
    axx[1,  1].set_ylabel('P value', fontsize=23)
    axx[1,  1].set_ylim(-0.03, 1.1)
    axx[1,  1].set_xlim(-10, 10)
    # axx[1,  1].set_aspect('equal', 'datalim')
    axx[1,  1].grid(axis='both')


    for i in range(2):
        for j in range(3):
            for xtick in axx[i, j].xaxis.get_major_ticks():
               xtick.label.set_fontsize(22)
            for ytick in axx[i, j].yaxis.get_major_ticks():
               ytick.label.set_fontsize(22)
            box = axx[i, j].get_position()
            axx[i, j].set_position([box.x0, box.y0+box.height*0.1,
                                   box.width*0.8, box.height*0.9])

    f.show()

    raw_input('press enter to quit')

if __name__ == '__main__':
    main()