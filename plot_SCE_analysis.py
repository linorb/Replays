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
    filename = 'SCE_analysis.npz'
    npzfile = np.load(filename)
    neurons_counter_all_mice = npzfile['neurons_counter_all_mice']
    count_run_all_mice = npzfile['count_run_all_mice']
    relevant_indices = npzfile['relevant_indices']

    box_data = divide_to_boxes(count_run_all_mice[relevant_indices],
                               neurons_counter_all_mice[relevant_indices])

    f, axx = subplots(3, 1)
    axx[0].hist(neurons_counter_all_mice[relevant_indices], normed=True)
    axx[0].set_title('Distribution of number of neurons per SCE', fontsize=18)
    axx[1].hist(count_run_all_mice[relevant_indices], normed=True)
    axx[1].set_title('Distribution of number of shared neurons in SCE and in '
                     'following run', fontsize=18)
    axx[2].boxplot(box_data)
    axx[2].set_title('number of neurons in SCE Vs.'
                     ' number of congruente neurons in run', fontsize=18)
    axx[2].set_xlabel('number of neurons in sce', fontsize=15)
    axx[2].set_ylabel('number of shared neurons \n in run and SCE', fontsize=15)
    axx[2].set_ylim(-0.5, 10)
    f.show()
    raw_input('press enter to quit')

if __name__ == '__main__':
    main()
