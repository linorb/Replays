import numpy as np
from matplotlib.pyplot import *

def divide_to_boxes(x,y):
    # divied x to boxes valued in y. assuming integers in y
    max_box_value = int(np.max(y))
    boxes = []
    for i in range(max_box_value):
        boxes.append(x[y == i])

    return boxes

def main():
    filename = r'results\SCE_analysis.npz'
    npzfile = np.load(filename)
    neurons_counter_all_mice = npzfile['neurons_counter_all_mice']
    count_run_all_mice = npzfile['count_run_all_mice']
    relevant_indices = npzfile['relevant_indices']

    box_data = divide_to_boxes(count_run_all_mice[relevant_indices],
                               neurons_counter_all_mice[relevant_indices])

    f, axx = subplots(3, 1)
    axx[0].hist(neurons_counter_all_mice[relevant_indices], normed=True)
    axx[0].set_xlabel('Number of neurons per SCE', fontsize=16)
    axx[0].set_ylabel('Density', fontsize=16)
    axx[1].hist(count_run_all_mice[relevant_indices], normed=True)
    axx[1].set_xlabel('Number of shared neurons in SCE and following run'
                      , fontsize=16)
    axx[1].set_ylabel('Density', fontsize=16)
    axx[2].boxplot(box_data)
    axx[2].set_xlabel('number of neurons in SCE', fontsize=16)
    axx[2].set_ylabel('number of shared neurons \n in SCE and following run',
                      fontsize=16)
    axx[2].set_ylim(-0.5, 10)
    axx[2].set_xticks(np.arange(0,40,5))
    axx[2].set_xticklabels(np.arange(0,40,5))

    for j in range(3):
        for xtick in axx[j].xaxis.get_major_ticks():
            xtick.label.set_fontsize(15)
        for ytick in axx[j].yaxis.get_major_ticks():
            ytick.label.set_fontsize(15)
        box = axx[j].get_position()
        axx[j].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    f.show()
    raw_input('press enter to quit')

if __name__ == '__main__':
    main()
