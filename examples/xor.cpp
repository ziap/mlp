#include <iostream>
#include <random>

#include "mlp.h"

int main() {
  constexpr size_t DATA_SIZE = 4;
  constexpr size_t INPUT_SIZE = 2;
  constexpr size_t OUTPUT_SIZE = 1;

  f32 input[DATA_SIZE * INPUT_SIZE] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
  };

  f32 output[DATA_SIZE * OUTPUT_SIZE] = {
    0,
    1,
    1,
    0
  };

  using Model = MLP<LossFn::MSE, INPUT_SIZE, 4, OUTPUT_SIZE>;
  f32 model_memory[Model::MEMORY_SIZE];

  std::random_device dev;
  Model model(dev(), model_memory);

  for (size_t i = 0; i < 10000; ++i) {
    model.Learn(DATA_SIZE, input, output, 1.0, 0.0);

    if ((i + 1) % 100 == 0) {
      std::cout << (i + 1) / 100 << ' ';
      std::cout << "Cost: " << model.Cost(DATA_SIZE, input, output) << '\n';
    }
  }

  std::cout << '\n';

  size_t correct = 0;

  for (size_t i = 0; i < DATA_SIZE; ++i) {
    f32 *predicted = model.Predict(input + i * INPUT_SIZE);
    f32 result = std::round(predicted[0]);

    if (result == output[i]) {
      correct ++;
    } else {
      std::cout << input[i * INPUT_SIZE] << '^' << input[i * INPUT_SIZE + 1];
      std::cout << " != " << result << '\n';
    }
  }

  std::cout << "Accuracy: " << 100.0f * correct / DATA_SIZE << "%\n";

  return 0;
}
