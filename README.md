# BboxDistances

BboxDistances is a tool that allows users to do compute the minimum distance between bounding boxes.

This is useful in object detection applications, where measuring the distance between objects (cars/people) is important. When the real size of objects is known (such as the width of cars, e.g. 6 ft), BboxDistances can measure 3D distances by constructing a depth image.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of `Python3`.
* You have installed `git`.
* You have a `MacOS` machine (not tested on `Linux/Windows`).

## Installing BboxDistances

To install BboxDistances, follow these steps:

```
git clone https://github.com/saadjansari/BboxDistances.git
```

BboxDistances uses `OpenCV`, `Numpy`, and `Scipy` libraries.

To set up the dependencies, create a virtual environment using venv. 

```
cd BboxDistances
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```


Additionally, to explore the demo in `jupyter`, add this kernel to ipython kernels:

```
python3 -m ipykernel install --user --name=env
```


## Using BboxDistances
We have 2 demos for this project, as well as a jupyter sandbox.

#### Single Image
To see an example of distance estimation between a pair of bounding boxes, run:
```
python3 main.py 0 ./examples/image1
```
This should open a grid of images showing measured distances (in feet).

#### Video
To see an example of distance estimation between a pair of bounding boxes in a video, run:
```
python3 main.py 1 ./examples/video2
```
After run is successful, a plot showing the measured distance over time will open.
An annotated movie showing measured distances is also saved here: `./examples/video2/movie_annotated.mp4`

#### Jupyter Sandbox
```
jupyter-notebook main_interactive.ipynb
```
Note: Before running cells, ensure that the `env` kernel (added to jupyter earlier) is selected.

## Contributors

Thanks to the following people who have contributed to this project:

* [@saadjansari](https://github.com/saadjansari) 📖


## Contact

If you want to contact me you can reach me at saadjansari@gmail.com.

## License

This project uses the following license: [MIT](https://opensource.org/licenses/MIT).
