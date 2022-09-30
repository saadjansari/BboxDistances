import numpy as np
import cv2
from .bbox_pair_distance import bbox_pair_distance

def draw_bbox_pair_distance(
    img, bb1, bb2, width1=1, width2=1, units="pixels", verbose=False
):
    # Draw bounding boxes and minimum distance vector on img

    # Draw bounding boxes and their numbers
    img_annotated = cv2.rectangle(
        img.copy(),
        bb1[:2].astype(np.uint32),
        bb1[2:].astype(np.uint32),
        color=(255, 0, 0),
        thickness=2,
    )
    img_annotated = cv2.putText(
        img_annotated,
        "1",
        10 + bb1[2:].astype(np.uint32),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    img_annotated = cv2.rectangle(
        img_annotated,
        bb2[:2].astype(np.uint32),
        bb2[2:].astype(np.uint32),
        color=(255, 0, 0),
        thickness=2,
    )
    img_annotated = cv2.putText(
        img_annotated,
        "2",
        10 + bb2[2:].astype(np.uint32),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # Find minimum distance between boxes
    dist, coords = bbox_pair_distance(
        bb1,
        bb2,
        img.shape[:2],
        width1=width1,
        width2=width2,
        units=units,
        verbose=verbose,
    )

    if coords is not None:
        coords_np = np.array(coords, dtype=np.uint32)

        # Draw minimum distance line
        img_annotated = cv2.line(
            img_annotated,
            coords_np[:2],
            coords_np[2:],
            color=(255, 255, 255),
            thickness=2,
        )

    # Write distance text
    text = f"{dist:.2f} {units}"
    x_text = 20
    y_text = img.shape[0] - 20
    img_annotated = cv2.putText(
        img_annotated,
        text,
        (x_text, y_text),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return img_annotated

if __name__ == '__main__':
    pass