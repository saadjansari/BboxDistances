import numpy as np
from .bbox_pair_distance import bbox_pair_distance


def bbox_distance_matrix(
    bbox,
    shape_img,
    widths=None,
    units="pixels",
    bbox_labels=None,
    verbose=False,
):
    """Minimum distance between all object bounding boxes.

    Args:
        bbox (np.array)     : shape (N, 4).
                            1st dimension N is the number of bounding boxes.
                            2nd dimension 4 contains the bounding box coordinates
                            (x_left, y_bottom, x_right, y_top)
        shape_img (np.array): 2D image shape [height, width]
        widths (np.array)   : Real width/size of objects (shape = (N,).
        units (str)         : 'pixel' or real units (default = 'pixel)
        bbox_labels (list)  : List of object labels (length = N)
        verbose (bool)      : log status (default = False)

    Returns:
        dist_matrix (np.array)  : shape (N,N).
                                element (i,j) contains the minimum distance
                                between the ith and the jth bounding box
    """

    # Number of objects
    num_bbox = bbox.shape[0]

    # If widths not specified, set to ones
    if widths is None:
        widths = np.ones(num_bbox)

    # Initalize distance matrix with -1
    dist_matrix = np.zeros((num_bbox, num_bbox), dtype=np.float32)
    dist_matrix[:] = -1

    # Loop over objects, calculate distance if not already calculated
    for i in np.arange(num_bbox):
        for j in np.arange(num_bbox):

            # If same object, set distance to 0
            if i == j:
                dist_matrix[i, j] = 0

            # If distance not calculated yet
            elif dist_matrix[i, j] == -1:

                # Find distance between these bounding boxes.
                dist_matrix[i, j], _ = bbox_pair_distance(
                    bbox[i, :],
                    bbox[j, :],
                    shape_img,
                    width1=widths[i],
                    width2=widths[j],
                    units=units,
                    verbose=False,
                )
                dist_matrix[j, i] = dist_matrix[i, j]

                # Print Distances
                if verbose:
                    print(
                        f"Object {i} ({bbox_labels[i]}) and Object {j} ({bbox_labels[j]}) are {dist_matrix[i,j]:.2f} {units} away."
                    )

            else:
                pass

    return dist_matrix