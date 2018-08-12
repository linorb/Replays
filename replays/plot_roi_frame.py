import cPickle
import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from zivlab.data_structs.tiff import load_tiff_file
from scipy.ndimage.filters import gaussian_filter

from bambi.analysis.roi import ROI
from bambi.tools.image import clipped_zoom
from bambi.tools.matlab import load_filters_matrix, load_log_file

MOUSE = '3'
CAGE = '40'
WORK_DIR = R'D:\dev\real_time_imaging_experiment_analysis\work_data_for_roi_figure'
if MOUSE == '3':
    ROIS_INDICES = [36, 53, 80, 89, 158, 181, 195, 229, 258, 290, 321, 336,
                         339, 357, 366, 392, 394, 399, 408, 439, 446, 448, 449,
                         465, 490]
else:
    ROIS_INDICES = [44, 61, 78, 96, 154, 157, 172, 195, 214, 226, 244, 247,
                     259, 261, 262, 286, 287, 290, 301, 303, 314, 337, 340,
                     346, 348, 368, 372, 374, 383, 389, 391, 407, 415, 418,
                     419, 448, 448, 460, 472, 473, 474, 479, 488, 501, 517, 569]

def create_rois_frame(rois):
    #  taken from bambi\bin\align_rois
    shape = rois[0].mask.shape
    frame = np.zeros(shape)

    for roi in rois:
        frame += roi.mask

    return frame

def create_microscope_frame(microscope_frames_filename):
    """Create a single microscope frame from the given file.

    If the file contains a single frame return the frame as is.
    If the file contains multiple frames, project maximum illumination
    per pixel of the DFoF frames and return the resulting frame.
    """
    # inspired from bambi\bin\align_rois
    microscope_frames = load_tiff_file(microscope_frames_filename)

    if microscope_frames.ndim == 2:
        return microscope_frames

    f0 = np.mean(microscope_frames, axis=0)

    dfof = (microscope_frames - f0)/f0
    return np.max(dfof, axis=0)


def align_rois_to_frame(microscope_frame, rois_frame, transformations,
                        cropping_coordinates):
    # OpenCV expect the data to be in the range [0,1] (for floats).
    # We convert from uint16 to floats and span the entire range.
    normalized_microscope_frame = microscope_frame / np.float(
        np.max(microscope_frame))
    normalized_rois_frame = rois_frame / np.float(np.max(rois_frame))

    # The following function handles only 32bit image files
    microscope_rgb_frame = cv2.cvtColor(
        normalized_microscope_frame.astype(np.float32),
        cv2.COLOR_GRAY2BGR)

    # cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
    if transformations == None:
        horizontal_shift = 0
        vertical_shift = 0
        zoom = 1
        rotation_angle = 0
        gamma = 1

        alpha = 1
    else:
        horizontal_shift = transformations['horizontal_shift']
        vertical_shift = transformations['vertical_shift']
        zoom = transformations['zoom']
        rotation_angle = transformations['rotation_angle']
        gamma = transformations['gamma']

        alpha = 1

    # Start from a fresh frame in every iteration
    rois_frame = normalized_rois_frame

    # Perform the transformations
    rois_frame = rotate(rois_frame, rotation_angle)
    rois_frame = clipped_zoom(rois_frame, zoom)
    # Use numpy roll since scipy's shift is slow.
    # Notice that the frame wraps around the screen.
    # The user shouldn't get to this state though so we can ignore
    # the issue.
    rois_frame = np.roll(rois_frame, vertical_shift, axis=0)
    rois_frame = np.roll(rois_frame, horizontal_shift, axis=1)

    # Adjust gamma to resize each individual ROI (high intensity pixels
    # will be kept high while low intensity would be decreased).
    rois_frame = adjust_gamma(rois_frame, gamma=gamma)

    # The following function handles only 32bit image files
    rois_rgb_frame = cv2.cvtColor(rois_frame.astype(np.float32),
                                  cv2.COLOR_GRAY2BGR)

    # Zero green and blue channels to make the ROIs red
    # rois_rgb_frame[:, :, 1:3] = 0

    rois_contour = cv2.Canny(np.uint8(rois_rgb_frame > 0), 0, 1)/255

    # rois_rgb_frame *= alpha
    rois_full_contour = cv2.cvtColor(
        rois_contour.astype(np.float32),
        cv2.COLOR_GRAY2BGR)
    rois_full_contour[:, :, 1:3] = 0
    rois_full_contour*= 0.5
    overlayed_frame = microscope_rgb_frame + rois_full_contour
    # plt.figure()
    # plt.imshow(overlayed_frame, aspect='auto')
    # plt.show()
    if cropping_coordinates:
        frame_to_show = overlayed_frame[cropping_coordinates[1]:cropping_coordinates[3],
                        cropping_coordinates[0]:cropping_coordinates[2]]
    else:
        frame_to_show = overlayed_frame
    return frame_to_show

def main():

    general_log_file = os.path.join(WORK_DIR, 'C%sM%s' %(CAGE, MOUSE),
                                'logFile.txt' )
    rois_filename = os.path.join(WORK_DIR, 'C%sM%s' % (CAGE, MOUSE),
                                 'finalFiltersMat.mat')
    fig, axx = plt.subplots(2, 3)
    for day in np.arange(0,6):
        day_dir = os.path.join(WORK_DIR, 'C%sM%s' %(CAGE, MOUSE), 'day%d' %day)
        microscope_frames_filename = os.path.join(day_dir, 'frames_norm.tif')
        transformations_filename = os.path.join(day_dir, 'transformations.pkl')
        local_log_file = os.path.join(day_dir, 'logFile.txt')

        rois_masks = load_filters_matrix(rois_filename)
        preprocessing_parameters = load_log_file(general_log_file)
        cropping_coordinates =  load_log_file(local_log_file)['Cropping coordinates']
        # To match python zero based numbering:
        cropping_coordinates = np.array(cropping_coordinates) - 1

        rois = []
        for i in ROIS_INDICES:
            rois.append(ROI(rois_masks[i], preprocessing_parameters))

        microscope_frame = create_microscope_frame(microscope_frames_filename)
        # filtered_microscope_frame = gaussian_filter(microscope_frame, 1)
        filtered_microscope_frame = microscope_frame

        rois_frame = create_rois_frame(rois)

        try:
            with open(transformations_filename, 'rb') as f:
                transformations = cPickle.load(f)
        except IOError:
            transformations = None

        current_frame = align_rois_to_frame(filtered_microscope_frame, rois_frame,
                            transformations, [])
        axx[day/3][day%3].imshow(current_frame, aspect='auto')
        axx[day/3][day%3].set_title('Session %d' %day)
        # axx[day/3][day%3].tick_params(axis='both', which='both', bottom='off', top='off',
        #                 labelbottom='off', labeltop='off')
        axx[day / 3][day % 3].axis('off')

    fig.suptitle('C%sM%s' %(CAGE, MOUSE), fontsize=20)

    fig.show()

    raw_input('Press enter')


if __name__ == '__main__':
    main()
