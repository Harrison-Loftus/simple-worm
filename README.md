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
./my_python --version
```
