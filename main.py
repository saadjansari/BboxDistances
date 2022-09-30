import os
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from BboxDistances.draw_bbox_pair_distance import draw_bbox_pair_distance
from BboxDistances.bbox_distances_multiframe import bbox_distances_multiframe
from BboxDistances.utils import read_video, save_video


def load_data(src_path):
 
    # Read image
    img = cv2.imread( os.path.join( src_path,'image.jpeg'))
    img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)

    # Read bounding boxes, class IDs, class_names
    with open( os.path.join(src_path,'bounding_boxes.npy'),'rb') as f:
        bounding_boxes = np.load(f)
    with open( os.path.join(src_path,'class_IDs.npy'),'rb') as f:
        class_ids = np.load(f)
    with open( os.path.join(src_path,'class_names.npy'),'rb') as f:
        class_names = np.load(f)
    
    return img, bounding_boxes, class_ids, class_names


def demo_image(src_path):
    # Demo for image

    # Load example data
    img, bounding_boxes, class_ids, class_names = load_data(src_path)


    # List for storing annotated images
    imgs = []

    # Test on first 6 objects
    for idx in range(0,6):
        imgs.append( 
            draw_bbox_pair_distance(
                img, 
                bounding_boxes[0,:], 
                bounding_boxes[idx,:],
                width1=6, width2=6,
                units='feet', verbose=True))

    # Show images
    fig = plt.figure(figsize=(12., 8.))
    grid = ImageGrid(fig, 111, 
            nrows_ncols=(3, 2),
            axes_pad=0.1,
            )
    for ax, im in zip(grid, imgs):
        ax.imshow(im)

    plt.show()


def demo_movie(src_path):
    # Demo for movie

    # Load sample movie
    imgs = read_video( os.path.join(src_path, 'movie.mp4'))

    # Load tracked bounding boxes
    with open( os.path.join(src_path,'bounding_boxes.npy'),'rb') as f:
        boxes = np.load(f)

    # Save annotated movie
    img_annot = []
    for frame in range(len(imgs)):
        img_annotated = draw_bbox_pair_distance(
            imgs[frame], 
            boxes[0,:,frame],
            boxes[1,:,frame],
            width1=6,
            width2=6,
            units='feet'
        )
        img_annot.append( img_annotated)
    save_video(img_annot,os.path.join(src_path,'movie_annotated.mp4'),fps=20)

    # Create distance plot
    dmat_real = bbox_distances_multiframe(boxes, imgs[0].shape[:2], widths=np.array([6,6]), units='feet')
    dmat_pix = bbox_distances_multiframe(boxes, imgs[0].shape[:2], widths=np.array([6,6]), units='pixels')
    fig,ax = plt.subplots(2,1, figsize=(8,6),sharex=True)

    ax[0].plot(dmat_real[0,1,:])
    ax[0].set_ylabel('Distance (feet)')
    ax[0].set_ylim(bottom=0)

    # make a plot with different y-axis using second axis object
    ax[1].plot(dmat_pix[0,1,:])
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel('Frame')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    args = sys.argv[1:]
    print(args)
    src_path = args[1]

    # single image test
    if int(args[0]) == 0:
        demo_image(src_path)

    # movie test
    elif int(args[0]) == 1:
        demo_movie(src_path)

