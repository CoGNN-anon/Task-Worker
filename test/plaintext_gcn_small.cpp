#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

template <typename T>
void print_vector_of_vector(const std::vector<std::vector<T>>& v, size_t lines = 0) {
  size_t cnt = 0;
  // Loop through each vector in the vector of vector
  for (auto vec : v) {
    if (lines != 0 && cnt >= lines) break;
    // Loop through each element in the vector
    for (auto elem : vec) {
      // Print the element with a space
      std::cout << elem << " ";
    }
    // Print a newline after each vector
    std::cout << "\n";
    cnt++;
  }
}

// Softmax function
std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& input) {
  // Get the number of rows and columns of the input matrix
  int n_rows = input.size();
  int n_cols = input[0].size();

  // Initialize the output matrix as a zero matrix
  std::vector<std::vector<double>> output(n_rows, std::vector<double>(n_cols, 0.0));

  // Loop over the rows of the input matrix
  for (int i = 0; i < n_rows; i++) {
    // Find the maximum value in the input row
    double max_value = input[i][0];
    for (int j = 1; j < n_cols; j++) {
      if (input[i][j] > max_value) {
        max_value = input[i][j];
      }
    }
    // Compute the sum of the exponentials of the input row
    double sum_exp = 0.0;
    for (int j = 0; j < n_cols; j++) {
      sum_exp += std::exp(input[i][j] - max_value);
    }
    // Compute the softmax values for the output row
    for (int j = 0; j < n_cols; j++) {
      output[i][j] = std::exp(input[i][j] - max_value) / sum_exp;
    }
  }

  // Return the output matrix
  return output;
}

// Forward function of GCN training
std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& W, const std::vector<std::vector<double>>& A, uint64_t layer = 0) {
  // Matrix multiplication of A and X
  std::vector<std::vector<double>> AX(X.size(), std::vector<double>(X[0].size(), 0.0));
  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X[0].size(); j++) {
      for (int k = 0; k < A[0].size(); k++) {
        AX[i][j] += A[i][k] * X[k][j];
      }
    }
  }
  // Matrix multiplication of AX and W
  std::vector<std::vector<double>> AXW(AX.size(), std::vector<double>(W[0].size(), 0.0));
  for (int i = 0; i < AX.size(); i++) {
    for (int j = 0; j < W[0].size(); j++) {
      for (int k = 0; k < AX[0].size(); k++) {
        AXW[i][j] += AX[i][k] * W[k][j];
      }
    }
  }

  std::vector<std::vector<double>> Z(AXW.size(), std::vector<double>(AXW[0].size(), 0.0));
  if (layer == 1) {
    Z = softmax(AXW);
    printf("prediction------\n");
    print_vector_of_vector(AXW);
    print_vector_of_vector(Z);
    printf("------\n");
  } else {
    // Activation function (ReLU)
    for (int i = 0; i < AXW.size(); i++) {
      for (int j = 0; j < AXW[0].size(); j++) {
        Z[i][j] = std::max(0.0, AXW[i][j]);
      }
    }
  }

  // Return the output matrix Z
  return Z;
}

// Backward function of GCN training
std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& W, const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& Z, const std::vector<std::vector<double>>& dZ, double lr, uint64_t layer = 0) {
  std::vector<std::vector<double>> AX(X.size(), std::vector<double>(X[0].size(), 0.0));
  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X[0].size(); j++) {
      for (int k = 0; k < A[0].size(); k++) {
        AX[i][j] += A[i][k] * X[k][j];
      }
    }
  }    

  printf("ah--------\n");
  print_vector_of_vector(AX);
  printf("--------\n");

  printf("p_minus_y--------\n");
  print_vector_of_vector(dZ);
  printf("--------\n");

  // Gradient of the activation function (ReLU)
  std::vector<std::vector<double>> dAXW(dZ.size(), std::vector<double>(dZ[0].size(), 0.0));
  if (layer == 0) {
    for (int i = 0; i < dZ.size(); i++) {
      for (int j = 0; j < dZ[0].size(); j++) {
        if (Z[i][j] > 0.0) {
          dAXW[i][j] = dZ[i][j];
        }
      }
    }
  } else {
    dAXW = dZ;
  }

  // Gradient of the matrix multiplication of AX and W
  std::vector<std::vector<double>> dW(W.size(), std::vector<double>(W[0].size(), 0.0));
  std::vector<std::vector<double>> dAX(AX.size(), std::vector<double>(AX[0].size(), 0.0));
  for (int i = 0; i < W.size(); i++) {
    for (int j = 0; j < W[0].size(); j++) {
      for (int k = 0; k < AX.size(); k++) {
        dW[i][j] += AX[k][i] * dAXW[k][j];
        dAX[k][i] += W[i][j] * dAXW[k][j];
      }
    }
  }
  // Gradient of the matrix multiplication of A and X
  std::vector<std::vector<double>> dX(X.size(), std::vector<double>(X[0].size(), 0.0));
  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X[0].size(); j++) {
      for (int k = 0; k < A.size(); k++) {
        dX[i][j] += A[k][i] * dAX[k][j];
      }
    }
  }
  // printf(">>>>>>>>>\n");
  // print_vector_of_vector(W);

  // // dW avg across samples
  // for (int i = 0; i < dW.size(); i++) {
  //   for (int j = 0; j < dW[0].size(); j++) {
  //     dW[i][j] = dW[i][j] / X.size();
  //   }
  // }

  printf("dW--------\n");
  print_vector_of_vector(dW);
  printf("--------\n");
  printf("W before--------\n");
  print_vector_of_vector(W);
  printf("--------\n");
  // Update the weight matrix W with gradient descent
  for (int i = 0; i < W.size(); i++) {
    for (int j = 0; j < W[0].size(); j++) {
      W[i][j] -= lr * dW[i][j];
    }
  }
  printf("W after--------\n");
  print_vector_of_vector(W);
  printf("--------\n");
  // print_vector_of_vector(W);
  // printf("<<<<<<<<<<\n");
  // Return dX
  return dX;
}

// Define the loss function (mean squared error)
double loss(std::vector<std::vector<double>> Y, std::vector<std::vector<double>> Y_pred) {
  double mse = 0.0;
  for (int i = 0; i < Y.size(); i++) {
    for (int j = 0; j < Y[0].size(); j++) {
      mse += std::pow(Y[i][j] - Y_pred[i][j], 2);
    }
  }
  mse /= (Y.size() * Y[0].size());
  return mse;
}

// Cross-entropy loss function
double cross_entropy_loss(std::vector<std::vector<double>> Y, std::vector<std::vector<double>> Y_pred) {
  // Get the number of nodes and classes
  int n_nodes = Y.size();
  int n_classes = Y[0].size();

  // Initialize the cross-entropy loss as zero
  double ce = 0.0;

  // Loop over the nodes and classes
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_classes; j++) {
      // Add the cross-entropy loss for each node and class
      ce += -Y[i][j] * std::log(Y_pred[i][j]);
    }
  }

  // Return the average cross-entropy loss
  return ce / n_nodes;
}

// Define the gradient of the loss function
std::vector<std::vector<double>> dloss(std::vector<std::vector<double>> Y, std::vector<std::vector<double>> Y_pred) {
  std::vector<std::vector<double>> dY_pred(Y.size(), std::vector<double>(Y[0].size(), 0.0));
  for (int i = 0; i < Y.size(); i++) {
    for (int j = 0; j < Y[0].size(); j++) {
      // dY_pred[i][j] = 2 * (Y_pred[i][j] - Y[i][j]) / (Y.size() * Y[0].size());
      dY_pred[i][j] = Y_pred[i][j] - Y[i][j];
    }
  }
  return dY_pred;
}

// Define the accuracy function (percentage of correct predictions)
double accuracy(std::vector<std::vector<double>> Y, std::vector<std::vector<double>> Y_pred) {
  int n_correct = 0;
  for (int i = 0; i < Y.size(); i++) {
    // Find the index of the maximum value in Y_pred[i]
    int max_index = 0;
    double max_value = Y_pred[i][0];
    for (int j = 1; j < Y[0].size(); j++) {
      if (Y_pred[i][j] > max_value) {
        max_index = j;
        max_value = Y_pred[i][j];
      }
    }
    // Check if the index matches the one-hot encoded label in Y[i]
    if (Y[i][max_index] == 1.0) {
      n_correct++;
    }
  }
  // Return the percentage of correct predictions
  return (double) n_correct / Y.size() * 100;
}

// Define the main function
int main() {
  // Set the random seed
  std::srand(42);

  // Define the number of nodes, features, and classes
  int n_nodes = 4;
  int n_features = 2;
  int n_classes = 3;

  // Define the vertex feature matrix X (randomly initialized)
  std::vector<std::vector<double>> X(n_nodes, std::vector<double>(n_features, 0.0));
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_features; j++) {
      X[i][j] = (double) std::rand() / RAND_MAX;
    }
  }

  printf(">>>> Feature matrix.\n");
  print_vector_of_vector(X);

  // Define the weighted adjacency matrix A (symmetric and normalized)
  std::vector<std::vector<double>> A(n_nodes, std::vector<double>(n_nodes, 0.0));
  A[0][1] = A[1][0] = 0.5;
  A[0][2] = A[2][0] = 0.5;
  A[1][3] = A[3][1] = 0.5;
  A[2][3] = A[3][2] = 0.5;
  
  A[0][0] = 1.0;
  A[1][1] = 1.0;
  A[2][2] = 1.0;
  A[3][3] = 1.0;

  // for (int i = 0; i < n_nodes; i++) {
  //   for (int j = 0; j < n_nodes && j <= i; ++j) {
  //       if ((i + j) % 4 == 0) {
  //           A[i][j] = 1.0;
  //           A[j][i] = 1.0;
  //       }
  //   }
  // }
  // print_vector_of_vector(A);
  // A = symnorm(A);
  // print_vector_of_vector(A);

  // Define the label matrix Y (one-hot encoded)
  std::vector<std::vector<double>> Y(n_nodes, std::vector<double>(n_classes, 0.0));
  // for (int i = 0; i < n_nodes; i++) {
  //   Y[i][std::rand() % n_classes] = 1.0;
  // }
  Y[0][0] = 1.0;
  Y[1][1] = 1.0;
  Y[2][2] = 1.0;
  Y[3][0] = 1.0;

  // Define the weight matrices W1 and W2 (randomly initialized)
  std::vector<std::vector<double>> W1(n_features, std::vector<double>(n_classes, 0.0));
  for (int i = 0; i < n_features; i++) {
    for (int j = 0; j < n_classes; j++) {
      // W1[i][j] = (double) std::rand() / RAND_MAX;
      W1[i][j] = 0.5;
    }
  }
  std::vector<std::vector<double>> W2(n_classes, std::vector<double>(n_classes, 0.0));
  for (int i = 0; i < n_classes; i++) {
    for (int j = 0; j < n_classes; j++) {
      // W2[i][j] = (double) std::rand() / RAND_MAX;
      W2[i][j] = 0.5;
    }
  }

  // Define the learning rate and the number of epochs
  double lr = 0.1;
  int n_epochs = 3;

  // Train the two-layer GCN
  for (int epoch = 0; epoch < n_epochs; epoch++) {
    // Forward pass
    std::vector<std::vector<double>> Z1 = forward(X, W1, A, 0); // First layer output
    std::vector<std::vector<double>> Z2 = forward(Z1, W2, A, 1); // Second layer output
    std::vector<std::vector<double>> Y_pred = Z2; // Final prediction

    // Compute the loss
    double los = cross_entropy_loss(Y, Y_pred);

    // Compute the accuracy
    double acc = accuracy(Y, Y_pred);

    // Print the loss and the accuracy
    std::cout << "Epoch " << epoch + 1 << ", Loss: " << los << ", Accuracy: " << acc << "%" << std::endl;

    // Backward pass
    std::vector<std::vector<double>> dY_pred = dloss(Y, Y_pred); // Gradient of the loss
    auto tmp = backward(Z1, W2, A, Z2, dY_pred, lr, 1); // Update W2
    std::vector<std::vector<double>> dZ1 = tmp; // Gradient of the first layer output
    backward(X, W1, A, Z1, dZ1, lr, 0); // Update W1
  }

  // Return 0
  return 0;
}