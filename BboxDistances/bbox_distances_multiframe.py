import numpy as np
from .bbox_distance_matrix import bbox_distance_matrix


def bbox_distances_multiframe(
    bbox, shape_img, widths=None, units="pixel", bbox_labels=None, verbose=False
):
    """Minimum distance between all object bounding boxes over time.

    Args:
        bbox (np.array): shape (N, 4, T).
                            1st dimension N is the number of bounding boxes.
                            2nd dimension 4 contains the bounding box coordinates
                            (x_left, y_bottom, x_right, y_top)
                            3rd dimension T is the number of frames
        shape_img (np.array): 2D image shape [height, width]
        widths (np.array)   : Real width/size of objects (shape = (N,).
        units (str)         : 'pixel' or real units (default = 'pixel)
        bbox_labels (list)  : List of object labels (length = N)
        verbose (bool)      : log status (default = False)

    Returns:
        dist_matrix (np.array)  : shape (N,N,T).
                                    element (i,j,t) contains the minimum distance
                                    between the ith and the jth bounding box
                                    at time t.
    """

    # Number of objects
    num_bbox = bbox.shape[0]

    # Number of frames
    num_frames = bbox.shape[2]

    # Initalize distance matrix with -1
    dist_matrix = np.zeros((num_bbox, num_bbox, num_frames), dtype=np.float32)
    dist_matrix[:] = -1

    # Loop over frames, calculating the distance matrix for each frame
    for jframe in np.arange(num_frames):
        dist_matrix[:, :, jframe] = bbox_distance_matrix(bbox[:, :, jframe], 
        shape_img, widths=widths, units=units, bbox_labels=bbox_labels, verbose=verbose)

    return dist_matrix