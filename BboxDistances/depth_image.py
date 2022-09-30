import numpy as np
from scipy.interpolate import interp1d


def depth_image(shape_img, bbox1, bbox2, width1, width2, verbose=False):
    """Generates an image showing distance scale factors (depth)

    Args:
        shape_img (np.array): 2D image shape [height, width]
        bbox1 (np.array)    : coordinates for first bounding box
                            (x_left, y_bottom, x_right, y_top)
        bbox2 (np.array)    : coordinates for second bounding box
                            (x_left, y_bottom, x_right, y_top)
        width1 (float)      : Real width/size of object 1 in given units
        width2 (float)      : Real width/size of object 2 in given units
        verbose (bool)      : log status (default = False)

    Returns:
        depthImage (2D np.array): Image/array showing distance scale factors
                                    or depth at different coordinates.
        mean_scale (float)  : real units per pixel - scaling factor
    """
    # Apparent width (in pixels)
    width1_apparent = bbox1[2] - bbox1[0]
    width2_apparent = bbox2[2] - bbox2[0]
    height1_apparent = bbox1[3] - bbox1[1]
    height2_apparent = bbox2[3] - bbox2[1]

    # y_top locations in image (these correspond to the width locations)
    y1 = bbox1[3]
    y2 = bbox2[3]

    # length scale at different y locations
    # scale1 = ((width1 / width1_apparent) + (5 / height1_apparent) )/2
    # scale2 = ((width2 / width2_apparent) + (5 / height2_apparent) )/2
    # scale1 = 5 / height1_apparent
    # scale2 = 5 / height2_apparent
    scale1 = width1 / width1_apparent
    scale2 = width2 / width2_apparent

    # Interpolate to create a depthImage
    f = interp1d([y1, y2], [scale1, scale2], fill_value="extrapolate")
    yscales = f(np.arange(0, shape_img[0]))
    depthImage = np.tile(yscales, (shape_img[1], 1)).transpose()

    mean_scale = (scale1 + scale2)/2

    # Display
    if verbose:
        print(f"Apparent Object Width = {[width1_apparent, width2_apparent]}")
        print(f"Y location = {[y1,y2]}")
        print(f"Scales = {[scale1, scale2]}")
        print(
            f"Scales From Image = {(depthImage[int(y1),100], depthImage[int(y2), 2])}"
        )

    return depthImage, mean_scale