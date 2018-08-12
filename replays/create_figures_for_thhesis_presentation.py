from matplotlib.pyplot import *

from bambi.tools.activity_loading import wide_binning
from bambi.tools.matlab import *

MVMT_FILENAME = r'D:\dev\replays\work_data\two_environments\c7m4\day1\envA\my_mvmt_smooth.mat'

def main():
    movement_data = load_mvmt_file(MVMT_FILENAME)
    bins = movement_data[2]['bin']
    bins = wide_binning(bins, 24, 2)
    top_edge = 10
    bottom_edge = 1

    f = figure()
    plot(bins)
    axhline(top_edge, color='red')
    axhline(bottom_edge, color='red')
    xlabel('Time [Sec]', fontsize=18)
    ylabel('Place bin', fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    f.show()
    raw_input()

if __name__ == '__main__':
    main()


