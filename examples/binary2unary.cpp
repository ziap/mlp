#include <iostream>
#include <random>

#include "../mlp.h"

int main(void) {
  constexpr size_t BITS = 7;
  constexpr size_t DATA_SIZE = 1 << BITS;

  constexpr size_t INPUT_SIZE = BITS;
  constexpr size_t OUTPUT_SIZE = 1 << BITS;

  f32 input[DATA_SIZE * INPUT_SIZE];
  f32 output[DATA_SIZE * OUTPUT_SIZE];

  for (size_t i = 0; i < DATA_SIZE; ++i) {
    for (size_t j = 0; j < INPUT_SIZE; ++j) {
      input[i * INPUT_SIZE + j] = (i >> j) & 1;
    }

    std::fill_n(output + i * OUTPUT_SIZE, OUTPUT_SIZE, 0);
    output[i * OUTPUT_SIZE + i] = 1;
  }

  using Model = MLP<LossFn::MSE, INPUT_SIZE, OUTPUT_SIZE>;
  f32 model_memory[Model::MEMORY_SIZE];

  std::random_device dev;
  Model model(dev(), model_memory);

  for (size_t i = 0; i < 100000; ++i) {
    model.FitBatch(DATA_SIZE, input, output, 1.0f, 0.0f);

    if ((i + 1) % 1000 == 0) {
      std::cout << (i + 1) / 1000 << ' ';
      std::cout << "Cost: " << model.Cost(DATA_SIZE, input, output) << '\n';
    }
  }

  std::cout << '\n';

  size_t correct = 0;

  for (size_t i = 0; i < DATA_SIZE; ++i) {
    f32 *predicted = model.Predict(input + i * INPUT_SIZE);
    f32 max = predicted[0];
    size_t max_idx = 0;

    for (size_t i = 1; i < OUTPUT_SIZE; ++i) {
      if (predicted[i] > max) {
        max = predicted[i];
        max_idx = i;
      }
    }

    if (i == max_idx) correct++;
    else std::cout << i << " != " << max_idx << '\n';
  }

  std::cout << "Accuracy: " << 100.0f * correct / DATA_SIZE << "%\n";

  return 0;
}
