import numpy as np
from matplotlib.pyplot import *

filename = 'SCE_analysis.npz'
npzfile = np.load(filename)
box_data = npzfile['box_data'].all()
neurons_counter_all_mice = npzfile['neurons_counter_all_mice'].all()
count_run_all_mice = npzfile['count_run_all_mice'].all()
relevant_indices = npzfile['relevant_indices'].all()


f, axx = subplots(3, 1)
axx[0].hist(neurons_counter_all_mice[relevant_indices], normed=True)
axx[0].set_title('number of neurons per SCE histogram')
axx[1].hist(count_run_all_mice[relevant_indices], normed=True)
axx[1].set_title('number of cells in SCE and following run')
axx[2].boxplot(box_data)
axx[2].set_title('number of neurons in SCE Vs.'
                 ' number of congruente neurons in run')
f.show()
raw_input('press enter to quit')