import numpy as np
from .depth_image import depth_image

def pixel2real_scale(shape_img, coords, bbox1, bbox2, width1, width2, verbose=False):
    """
    Scale factor for converting distance to real units using object size -> depth reasoning.

    Args:
        shape_img (np.array): 2D image shape [height, width]
        coords (tuple)      : coordinates of minimum distance vector from
                            bbox1 to bbox2
                            (x1, y1, x2, y2)
        bbox1 (np.array)    : coordinates for first bounding box
                            (x_left, y_bottom, x_right, y_top)
        bbox2 (np.array)    : coordinates for second bounding box
                            (x_left, y_bottom, x_right, y_top)
        width1 (float)      : Real width/size of object 1 in given units
        width2 (float)      : Real width/size of object 2 in given units
        verbose (bool)      : log status (default = False)

    Returns:
        mean_scale (float)  : Scale factor for converting pixels into other units.
    """

    # Find depth image
    depthImage, mean_scale = depth_image(shape_img, bbox1, bbox2, width1, width2, verbose=verbose)

    # Find scale at minimum distance vector coordinates
    scale1 = depthImage[int(coords[1]), int(coords[0])]
    scale2 = depthImage[int(coords[3]), int(coords[2])]

    # Mean scale
    # mean_scale = 0.5 * (scale1 + scale2)

    return mean_scale