# Fashion-MNIST with convolutional networks and transfer learning

This repo contains code for training a number of different CNN models using the Fashion-MNIST dataset.
It also includes training a model on top of Inception-ResNet-v2 and doing gridsearch to find optimal training parameters.

## Pre-requisites

* Docker >= 18.06.1

Optional for running without docker:

* Python 3 with the following dependencies installed:
    * tensorflow 1.12
    * numpy
    * pandas
    * scikit-learn
    * matplotlib
    * scipy

## How to run

Using docker:

```
docker-compose up
```

Not using docker:

```
juputer-lab
```

This will start JupyterLab at http://localhost:8888/. Copy the URL including the token from the shell output to automatically log in.

- Open `Fashion-MNIST-CNN.ipynb`.
- Hit CMD-SHIFT-C to bring up the console palette.
- Choose "Run all cells"

Please note that it will take a while (minutes probably) to run all the preprocessing and training and. If you are using Docker for Mac or Windows, there may not be enough memory for your VM to do the preprocessing. In that case, I recommend increasing the memory available to the VM or by installing JupyterLab directly on your OS.

## Run grid search in the terminal

`python run_grid_search.py`

This can be run on a GPU instance on FloydHub (you need an account and the cli tool) by using the command:

`floyd filip$ floyd run --gpu --env tensorflow-1.12 "python run_grid_search.py"`

See https://docs.floydhub.com/getstarted/quick_start/ for info on how to get started using Floyd.
