#ifndef MLP_H
#define MLP_H

#include <cstdint>
#include <cstddef>

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

#ifndef MLP_MEMSET
#define MLP_MEMSET __builtin_memset
#endif

#ifndef MLP_EXP
#define MLP_EXP __builtin_expf
#endif

// 32-bit PCG with 64-bit state
// See: <https://www.pcg-random.org>
class Rng {
public:
  Rng(u64 seed) {
    state = INC + seed;
    state = state * MUL + INC;
  }

  u32 raw() {
    u64 oldstate = state;
    state = oldstate * MUL + INC;

    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  f32 rand() {
    return (f32)raw() / ((f32)(-1u) + 1.0f);
  }
private:
  static constexpr u64 MUL = 6364136223846793005ULL;
  static constexpr u64 INC = 1442695040888963407ULL;

  u64 state;
};

namespace ActivationFn {

class Sigmoid {
public:
  static f32 Compute(f32 x) {
    return 1.0f / (1.0f + MLP_EXP(-x));
  }

  static f32 Derivative(f32 x) {
    f32 sigmoid = Compute(x);

    return sigmoid * (1.0f - sigmoid);
  }
};

class ReLU {
public:
  static f32 Compute(f32 x) {
    return x < 0 ? 0 : x;
  }

  static f32 Derivative(f32 x) {
    return x < 0 ? 0 : 1;
  }
};

// TODO: Softmax

}

namespace LossFn {

class MSE {
public:
  static f32 Compute(f32 value, f32 expected) {
    f32 error = value - expected;

    return error * error;
  }

  static f32 Derivative(f32 value, f32 expected) {
    return 2.0f * (value - expected);
  }
};

// TODO: Categorical cross-entropy

}

using Layer = struct {
  f32 *parameters;
  f32 *sum;
  f32 *activations;

  f32 *parameters_grad;
  f32 *sum_grad;
};

template<class Loss, size_t N1, size_t N2, size_t ...Ns>
class MLPArch {
public:
  static constexpr size_t N_IN = N1;
  static constexpr size_t N_OUT = N2;

  static constexpr size_t INPUT_SIZE = N_IN;
  static constexpr size_t OUTPUT_SIZE = MLPArch<Loss, N_OUT, Ns...>::OUTPUT_SIZE;
  static constexpr size_t NODES_SIZE = MLPArch<Loss, N_IN, N_OUT>::NODES_SIZE + MLPArch<Loss, N_OUT, Ns...>::NODES_SIZE;
  static constexpr size_t PARAMS_SIZE = MLPArch<Loss, N_IN, N_OUT>::PARAMS_SIZE + MLPArch<Loss, N_OUT, Ns...>::PARAMS_SIZE;

  // Forward pass without computing the gradient
  static f32* Forward(f32 *input, Layer current_layer) {
    // Compute the layer's outputs
    f32 *activations = MLPArch<Loss, N_IN, N_OUT>::Forward(input, current_layer);

    // Send the output to the next layer
    return MLPArch<Loss, N_OUT, Ns...>::Forward(activations, NextLayer(current_layer));
  }

  // Feed the network and compute the gradient
  static size_t BackPropagate(f32 *input, f32 *output, Layer current_layer) {
    // Feed the current layer
    f32 *activations = MLPArch<Loss, N_IN, N_OUT>::Forward(input, current_layer);
    Layer next_layer = NextLayer(current_layer);

    // Recursively perform back-propagation on the next layer
    size_t next_size = MLPArch<Loss, N_OUT, Ns...>::BackPropagate(activations, output, next_layer);

    f32 *weights = next_layer.parameters;
    for (size_t i = 0; i < N_OUT; ++i) {

      f32 acc = 0;
      for (size_t j = 0; j < next_size; ++j) acc += weights[j * N_OUT + i] * next_layer.sum_grad[j];
      
      f32 activation_derivative = ActivationFn::Sigmoid::Derivative(current_layer.sum[i]);

      current_layer.sum_grad[i] = acc * activation_derivative;
    }

    f32 *weights_derivative = current_layer.parameters_grad;
    f32 *biases_derivative = current_layer.parameters_grad + N_IN * N_OUT;

    for (size_t i = 0; i < N_OUT; ++i) {
      for (size_t j = 0; j < N_IN; ++j) weights_derivative[i * N_IN + j] += input[j] * current_layer.sum_grad[i];

      biases_derivative[i] += current_layer.sum_grad[i];
    }

    return N_OUT;
  }

  static Layer NextLayer(Layer current_layer) {
    current_layer.parameters += N_IN * N_OUT + N_OUT;
    current_layer.sum += N_OUT;
    current_layer.activations += N_OUT;

    current_layer.parameters_grad += N_IN * N_OUT + N_OUT;
    current_layer.sum_grad += N_OUT;

    return current_layer;
  }
};

template<class Loss, size_t N1, size_t N2>
class MLPArch<Loss, N1, N2> {
public:
  static constexpr size_t N_IN = N1;
  static constexpr size_t N_OUT = N2;

  static constexpr size_t INPUT_SIZE = N_IN;
  static constexpr size_t OUTPUT_SIZE = N_OUT;
  static constexpr size_t NODES_SIZE = N_OUT;
  static constexpr size_t PARAMS_SIZE = N_IN * N_OUT + N_OUT;

  // Base case of recursive forward pass with a single layer
  static f32* Forward(f32 *input, Layer current_layer) {
    f32 *weights = current_layer.parameters;
    f32 *biases = current_layer.parameters + N_IN * N_OUT;

    // TODO: Optimize with SIMD
    for (size_t i = 0; i < N_OUT; ++i) {
      // W * x
      f32 acc = 0;
      for (size_t j = 0; j < N_IN; ++j) acc += weights[i * N_IN + j] * input[j]; 

      // W * x + b
      current_layer.sum[i] = acc + biases[i];
      
      // sigmoid(W * x + b)
      current_layer.activations[i] = ActivationFn::Sigmoid::Compute(current_layer.sum[i]);
    }

    return current_layer.activations;
  }

  // Base case of recursive back-propagation with the outer layer
  static size_t BackPropagate(f32 *input, f32 *output, Layer current_layer) {
    Forward(input, current_layer);

    for (size_t i = 0; i < N_OUT; ++i) {
      f32 cost_derivative = Loss::Derivative(current_layer.activations[i], output[i]);
      f32 activation_derivative = ActivationFn::Sigmoid::Derivative(current_layer.sum[i]);

      current_layer.sum_grad[i] = cost_derivative * activation_derivative;
    }
    
    f32 *weights_grad = current_layer.parameters_grad;
    f32 *biases_grad = current_layer.parameters_grad + N_IN * N_OUT;

    for (size_t i = 0; i < N_OUT; ++i) {
      for (size_t j = 0; j < N_IN; ++j) weights_grad[i * N_IN + j] += input[j] * current_layer.sum_grad[i];

      biases_grad[i] += current_layer.sum_grad[i];
    }

    return N_OUT;
  }
};

template<class Loss, size_t ...N>
class MLP {
public:
  using Arch = MLPArch<Loss, N...>;

  static constexpr size_t MEMORY_SIZE = Arch::PARAMS_SIZE * 3 + Arch::NODES_SIZE * 3;

  MLP(u64 seed, f32 *data) {
    layer.parameters = data;
    data += Arch::PARAMS_SIZE;

    layer.sum = data;
    data += Arch::NODES_SIZE;

    layer.activations = data;
    data += Arch::NODES_SIZE;

    layer.parameters_grad = data;
    data += Arch::PARAMS_SIZE;

    layer.sum_grad = data;
    data += Arch::NODES_SIZE;

    accumulated_gradient = data;

    Rng rng(seed);
    for (size_t i = 0; i < Arch::PARAMS_SIZE; ++i) {
      layer.parameters[i] = rng.rand();
    }

    MLP_MEMSET(accumulated_gradient, 0, Arch::PARAMS_SIZE * sizeof(f32));
  }

  f32* Predict(f32 *input) {
    return Arch::Forward(input, layer);
  }

  f32 Cost(size_t size, f32 *inputs, f32 *outputs) {
    f32 acc = 0;
    for (size_t i = 0; i < size; ++i) {

      f32 *predicted = Predict(inputs + (i * Arch::INPUT_SIZE));

      for (size_t j = 0; j < Arch::OUTPUT_SIZE; ++j) {
        f32 error = predicted[j] - outputs[i * Arch::OUTPUT_SIZE + j];
        acc += error * error; 
      }
    }

    return acc / (f32)size;
  }

  void FitBatch(size_t batch_size, f32 *batch_inputs, f32 *batch_outputs, f32 rate, f32 momentum) {
    MLP_MEMSET(layer.parameters_grad, 0, Arch::PARAMS_SIZE * sizeof(f32));

    // TODO: Process multiple data points in parallel
    for (size_t i = 0; i < batch_size; ++i) {
      f32 *input = batch_inputs + Arch::N_IN * i;
      f32 *output = batch_outputs + Arch::OUTPUT_SIZE * i;

      Arch::BackPropagate(input, output, layer);
    }

    for (size_t i = 0; i < Arch::PARAMS_SIZE; ++i) {
      accumulated_gradient[i] = accumulated_gradient[i] * momentum + layer.parameters_grad[i];

      layer.parameters[i] -= accumulated_gradient[i] * rate / (f32)batch_size;
    }
  }

private:
  Layer layer;

  f32 *accumulated_gradient;
};

#endif
