import numpy as np
from matplotlib.pyplot import *

def divide_to_boxes(x,y):
    # divied x to boxes valued in y. assuming integers in y
    max_box_value = int(np.max(y))
    boxes = []
    for i in range(max_box_value):
        boxes.append(x[y == i])

    return boxes

def add_subplot_axes(ax,rect,axisbg='w'):
    # Taken from: https://stackoverflow.com/questions/17458580/embedding-small
    # -plots-inside-subplots-in-matplotlib
    fig = gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def main():

    filename = r'results\SCE_analysis_Linear.npz'
    npzfile = np.load(filename)
    neurons_counter_all_mice = npzfile['neurons_counter_all_mice']
    count_run_all_mice = npzfile['count_run_all_mice']
    relevant_indices = npzfile['relevant_indices']

    box_data = divide_to_boxes(count_run_all_mice[relevant_indices],
                               neurons_counter_all_mice[relevant_indices])

    f, axx = subplots(3, 2, sharey='row', sharex='row')
    axx[0, 0].hist(neurons_counter_all_mice[relevant_indices], normed=True)
    axx[0, 0].set_xlabel('Number of neurons per SCE', fontsize=25)
    axx[0, 0].set_ylabel('Density', fontsize=25)
    axx[0, 0].set_title('Environment A', fontsize=25)
    rect = [0.6, 0.6, 0.3, 0.3]
    insetA = add_subplot_axes(axx[0, 0], rect)
    insetA.hist(neurons_counter_all_mice[relevant_indices], normed=True)
    insetA.set_yscale('log')
    axx[1, 0].hist(count_run_all_mice[relevant_indices], normed=True)
    axx[1, 0].set_xlabel('Number of shared neurons in SCE and following run'
                      , fontsize=25)
    axx[1, 0].set_ylabel('Density', fontsize=25)
    insetB = add_subplot_axes(axx[1, 0], rect)
    insetB.hist(count_run_all_mice[relevant_indices], normed=True)
    insetB.set_yscale('log')
    axx[2, 0].boxplot(box_data)
    axx[2, 0].set_xlabel('number of neurons in SCE', fontsize=25)
    axx[2, 0].set_ylabel('number of shared neurons \n in SCE and following run',
                      fontsize=25)
    axx[2, 0].set_xticks(np.arange(0,40,5))
    axx[2, 0].set_xticklabels(np.arange(0,40,5))

    filename = r'results\SCE_analysis_Lshape.npz'
    npzfile = np.load(filename)
    neurons_counter_all_mice = npzfile['neurons_counter_all_mice']
    count_run_all_mice = npzfile['count_run_all_mice']
    relevant_indices = npzfile['relevant_indices']

    box_data = divide_to_boxes(count_run_all_mice[relevant_indices],
                               neurons_counter_all_mice[relevant_indices])

    axx[0, 1].hist(neurons_counter_all_mice[relevant_indices], normed=True)
    axx[0, 1].set_xlabel('Number of neurons per SCE', fontsize=25)
    axx[0, 1].set_title('Environment B', fontsize=25)
    rect = [0.6, 0.6, 0.3, 0.3]
    insetA = add_subplot_axes(axx[0, 1], rect)
    insetA.hist(neurons_counter_all_mice[relevant_indices], normed=True)
    insetA.set_yscale('log')
    axx[1, 1].hist(count_run_all_mice[relevant_indices], normed=True)
    axx[1, 1].set_xlabel('Number of shared neurons in SCE and following run'
                      , fontsize=25)
    insetB = add_subplot_axes(axx[1, 1], rect)
    insetB.hist(count_run_all_mice[relevant_indices], normed=True)
    insetB.set_yscale('log')
    axx[2, 1].boxplot(box_data)
    axx[2, 1].set_xlabel('number of neurons in SCE', fontsize=25)
    axx[2, 1].set_xticks(np.arange(0,40,5))
    axx[2, 1].set_xticklabels(np.arange(0,40,5))

    for j in range(3):
        for i in range(2):
            for xtick in axx[j, i].xaxis.get_major_ticks():
                xtick.label.set_fontsize(22)
            for ytick in axx[j, i].yaxis.get_major_ticks():
                ytick.label.set_fontsize(22)
            box = axx[j, i].get_position()
            axx[j, i].set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])
    f.show()
    raw_input('press enter to quit')

if __name__ == '__main__':
    main()
