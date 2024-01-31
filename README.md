# Multilayer perceptron - Academic project

A multilayer perceptron implementation in Python.

To start off, clone this branch of the repo into your local:

```shell
git clone https://github.com/aL0NEW0LF/multilayer-perceptron-py
```

After cloning the project, create your virutal environment:

```shell
cd multilayer-perceptron-py
```

**Windows**

```shell
py -3 -m venv .venv
```

**MacOS/Linus**

```shell
python3 -m venv .venv
```

Then, activate the env:

**Windows**

```shell
.venv\Scripts\activate
```

**MacOS/Linus**

```shell
. .venv/bin/activate
```

You can run the following command to install the dependencies:

```shell
pip3 install -r requirements.txt
```

Then you are good to go.

# TODO

- [X] Implement it _KEKW_
- [X] Choose the number of hidden layers to have, choose the number of neurons in each hidden layer and activation function in each layer
- [ ] Fix the `RuntimeWarning: overflow encountered in exp` bug/runtime-error. (Somewhere in weight or error computing probably, or initialization, or some wrong sign) 

    Its the Exploding Gradients problem.
    
    An error gradient is the direction and magnitude obtained during neural network training and used to update network weights in the appropriate direction and quantity.

    Error gradients in deep or recurrent neural networks can collect during an update, resulting in extremely high gradients. These causes significant modifications to the network weights, resulting in an unstable network. Weight values can get so big that they overflow, resulting in NaN values.

    The explosion happens as a result of exponential growth, which is achieved by continuously multiplying gradients with values greater than 1.0 across network layers.

# Contributing guidelines

To contribute, see [contributing guidelines](CONTRIBUTING.md).
