# Getting started

- Clone this repository to your system (Copy all the files).

## Dependencies using conda (linux or macos):

- Install [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main).

- Create a conda environment:

  ```sh
$ conda env create -f environment.yml
  ```

## Running code using conda (linux or macos):

- Activate your conda environment:

``` sh
$ conda env activate simple-worm-scripts
```

Then run your code with Python (e.g.):

``` sh
$ python forward_model.py
```

## Running code using docker (any system):

- Install [`docker`](https://www.docker.com/) on your system.

- Use the helper script to run your code within the docker container (e.g.):

``` sh
./my_python forward_model.py
```

# Attributation

If you use this code please cite:

| Thomas Ranner . A stable finite element method for low inertia undulatory locomotion in three dimensions. Applied Numerical Mathematics 156 (2020) 422–44, 2020. https://doi.org/10.1016/j.apnum.2020.05.009

# Usage

The main interface is provided by the `Worm` class in `worm.py`. It is initialized with two arguments which specifies how many mesh points to use and the time step. It is recommended to use around 100 points (`N=101`) for accurate simulations and a time step around 1/1000 (`dt=1e-3`).

After initialization the method `update` assembles and solves the matrices with its argument used as the internal forcing. This is the prescribed curvature for the above paper. Currently this is only implemented for `numpy` arrays.

The output of the `Worm` object is summarized as follows:

- `x` :: positions of mid-line points.
- (`e1`, `e2`) :: components of the orthogonormal frame attached to the mid-line.
- (`alpha`, `beta`, `gamma`) :: an intrinsic representation of the body. `alpha` is curvature in the `e1` direction, `beta` is curvature in the `e2` direction and `gamma` describes the twist of the frame about the mid-line.

