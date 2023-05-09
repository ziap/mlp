# MLP

Multi-layer perceptron a.k.a Neural Network library.

A basic neural network framework from scratch in C++ without any 3rd-party
dependencies.

The goal is to create a model powerful enough to solve the [handwritten digit
classification problem][1] with reasonable training time and test accuracy.

Progress:

- [x] Neural network with configurable architecture
- [x] Gradient descent using backpropagation
- [x] Gradient descent with [momentum][2]
- [ ] Hardware acceleration (SIMD, multi-threading)
- [ ] Dataset processing (one-hot encoding/decoding, shuffling, batching, ...)
- [ ] Data and model serialization
- [ ] Metrics for model evaluation
- [ ] Output layer softmax activation
- [ ] Categorical cross-entropy loss function
- [ ] Better weight initialization
- [ ] [Adam][3] optimizer

## Usage

Copy the [mlp.h](mlp.h) header file to your working directory.

Look at the examples and the source code if you want to learn more.

### Building the examples

You will need a C++ compiler that supports C++11.

```bash
./build.sh
```

To build with g++:

```bash
CXX=g++ ./build.sh
```

## License

This project is licensed under the [MIT License](LICENSE).

[//]: # (References)
[1]: <https://en.wikipedia.org/wiki/MNIST_database>
[2]: <https://optimization.cbe.cornell.edu/index.php?title=Momentum>
[3]: <https://optimization.cbe.cornell.edu/index.php?title=Adam>
