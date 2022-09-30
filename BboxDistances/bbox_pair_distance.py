import numpy as np
from .pixel2real_scale import pixel2real_scale

def bbox_pair_distance(
    bbox1, bbox2, shape_img=None, units="pixel", width1=1, width2=1, verbose=False
):
    """Minimum distance between a pair of bounding boxes.

    Finds the pair of vertices on the bounding boxes that are closest to each other,
    giving a minimum vector. Then computes the euclidean distance between the boxes in
    pixels, given by the length of the vector.
    By default, it finds the distance in pixels.
    If units is not 'pixel', then it computes a 3D distance by estimating depth.
    This estimation assumes that the real size / width or the objects is known.
    Note: If the bounding boxes overlap, the distance is 0.

    Args:
        bbox1 (np.array)    : coordinates for first bounding box
                            (x_left, y_bottom, x_right, y_top)
        bbox2 (np.array)    : coordinates for second bounding box
                            (x_left, y_bottom, x_right, y_top)
        shape_img (np.array): 2D image shape [height, width]
        units (str)         : 'pixel' or real units (default = 'pixel)
        width1 (float)      : Real width/size of object 1 in given units
        width2 (float)      : Real width/size of object 2 in given units
        verbose (bool)      : log status (default = False)

    Returns:
        dist (float)   : minimum distance between bounding boxes (in pixels)
        coords (tuple) : coordinates of minimum distance vector from bb1 to bb2
                        (x1, y1, x2, y2)
    """

    # Unpack coordinates for simplicity
    x1_left, y1_bottom, x1_right, y1_top = bbox1
    x2_left, y2_bottom, x2_right, y2_top = bbox2

    # Find bounding box 2 general location relative to bounding box 1
    is_left = x2_right < x1_left
    is_right = x2_left > x1_right
    is_top = y2_bottom > y1_top
    is_bottom = y2_top < y1_bottom

    # Display position status
    if verbose:
        if is_left:
            print("Bounding Box 2 is LEFT of Bounding Box 1")
        if is_right:
            print("Bounding Box 2 is RIGHT of Bounding Box 1")
        if is_top:
            print("Bounding Box 2 is ABOVE Bounding Box 1")
        if is_bottom:
            print("Bounding Box 2 is BELOW Bounding Box 1")

    # Tests
    assert not (
        is_left and is_right
    ), "Bounding Box 2 cannot be both on the left and the right of Bounding Box 1"
    assert not (
        is_top and is_bottom
    ), "Bounding Box 2 cannot be both above and below Bounding Box 1"

    # Calculate Minimum Distance based on position
    # Here, we find the coordinates of the minimum distance vector
    # If top and left, then mimimum distance vector goes from bottom right vertex of bb2 and top left vertex of bb1
    if is_top and is_left:
        coords = (x1_left, y1_top, x2_right, y2_bottom)

    # If top and right, then mimimum distance vector goes from bottom left vertex of bb2 and top right vertex of bb1
    elif is_top and is_right:
        coords = (x1_right, y1_top, x2_left, y2_bottom)

    # If bottom and right, then mimimum distance vector goes from top left vertex of bb2 and bottom right vertex of bb1
    elif is_bottom and is_right:
        coords = (x1_right, y1_bottom, x2_left, y2_top)

    # If bottom and left, then mimimum distance vector goes from top right vertex of bb2 and bottom left vertex of bb1
    elif is_bottom and is_left:
        coords = (x1_left, y1_bottom, x2_right, y2_top)

    # If only bottom, then minimum distance vector goes from top of bb2 and bottom of bb1
    elif is_bottom:
        xc = np.mean(sorted([x1_left, x1_right, x2_left, x2_right])[1:3])
        coords = (xc, y1_bottom, xc, y2_top)

    # If only top, then minimum distance vector goes from bottom of bb2 and top of bb1
    elif is_top:
        xc = np.mean(sorted([x1_left, x1_right, x2_left, x2_right])[1:3])
        coords = (xc, y1_top, xc, y2_bottom)

    # If only left, then minimum distance vector goes from right of bb2 and left of bb1
    elif is_left:
        yc = np.mean(sorted([y1_bottom, y1_top, y2_bottom, y2_top])[1:3])
        coords = (x1_left, yc, x2_right, yc)

    # If only right, then minimum distance vector goes from left of bb2 and right of bb1
    elif is_right:
        yc = np.mean(sorted([y1_bottom, y1_top, y2_bottom, y2_top])[1:3])
        coords = (x1_right, yc, x2_left, yc)

    # Otherwise, bounding boxes overlap, and minimum distance is 0
    else:
        coords = None

    if coords is not None:
        coords = (coords[0], y1_top, coords[2], y2_top)

    # Calculate L2 distance of coords
    dist = euc_dist(*coords) if coords is not None else 0

    # Convert distance to other units if required:
    if (dist != 0) and (units != "pixels"):

        # Get pixel -> real-units scale factor
        scale = pixel2real_scale(shape_img, coords, bbox1, bbox2, width1, width2, verbose=verbose)

        # Apply scaling factor to distance
        dist *= scale

    # Display
    if verbose:
        print(f"Distance = {dist:.2f} {units}")

    return dist, coords

def euc_dist(x1, y1, x2, y2):
    # Euclidean L2 distance between 2 points: (x1,y1) and (x2,y2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
