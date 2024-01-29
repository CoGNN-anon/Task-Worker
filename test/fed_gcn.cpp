#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <map>

// // >> Cora
// // Define the number of nodes, features, and classes
// int n_nodes = 2708;
// int n_features = 1433;
// int n_classes = 7;
// int n_hidden = 16;
// double train_ratio = 0.2;
// double val_ratio = 0.2;
// double test_ratio = 0.6;
// // Define the learning rate and the number of epochs
// double lr = 0.5;
// int n_epochs = 90;
// int n_parts = 2;
// std::string v_path = "/home/zzh/project/SecGNN/data/Cora/transformed/cora.vertex.preprocessed";
// std::string e_path = "/home/zzh/project/SecGNN/data/Cora/transformed/cora.edge.preprocessed";
// std::string p_path = std::string("/home/zzh/project/SecGNN/data/Cora/transformed/cora.part.preprocessed.") + std::to_string(n_parts) + std::string("p");
// std::string dataset = "cora";
// std::string Dataset = "Cora";

// // >> citeseer
// // Define the number of nodes, features, and classes
// int n_nodes = 3327;
// int n_features = 3703;
// int n_classes = 6;
// int n_hidden = 16;
// // Define the learning rate and the number of epochs
// double lr = 0.4;
// double train_ratio = 0.2;
// double val_ratio = 0.2;
// double test_ratio = 0.6;
// int n_epochs = 90;
// int n_parts = 2;
// std::string v_path = "/home/zzh/project/SecGNN/data/Citeseer/transformed/citeseer.vertex.preprocessed";
// std::string e_path = "/home/zzh/project/SecGNN/data/Citeseer/transformed/citeseer.edge.preprocessed";
// std::string p_path = std::string("/home/zzh/project/SecGNN/data/Citeseer/transformed/citeseer.part.preprocessed.") + std::to_string(n_parts) + std::string("p");
// std::string dataset = "citeseer";
// std::string Dataset = "Citeseer";

// >> Pubmed
// Define the number of nodes, features, and classes
int n_nodes = 19717;
int n_features = 500;
int n_classes = 3;
int n_hidden = 16;
// Define the learning rate and the number of epochs
double lr = 8.0;
double train_ratio = 0.05;
double val_ratio = 0.15;
double test_ratio = 0.8;
int n_epochs = 90;
int n_parts = 2;
std::string v_path = "/home/zzh/project/SecGNN/data/Pubmed/transformed/pubmed.vertex.preprocessed";
std::string e_path = "/home/zzh/project/SecGNN/data/Pubmed/transformed/pubmed.edge.preprocessed";
std::string p_path = std::string("/home/zzh/project/SecGNN/data/Pubmed/transformed/pubmed.part.preprocessed.") + std::to_string(n_parts) + std::string("p");
std::string dataset = "pubmed";
std::string Dataset = "Pubmed";

// Matrix multiplication of A and B
std::vector<std::vector<double>> matmul(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B) {
  // Get the number of rows and columns of the matrices
  int n_rows = A.size();
  int n_cols = B[0].size();
  int n_inner = A[0].size();

  // Initialize the output matrix C as a zero matrix
  std::vector<std::vector<double>> C(n_rows, std::vector<double>(n_cols, 0.0));

  // Set the number of threads for OMP
  int n_threads = 32; // You can change this according to your system
  omp_set_num_threads(n_threads);

  // // Parallelize the matrix multiplication with OMP
  // #pragma omp parallel for
  // for (int i = 0; i < n_rows; i++) {
  //   for (int j = 0; j < n_cols; j++) {
  //     // Loop over the inner dimension of the matrices
  //     for (int k = 0; k < n_inner; k++) {
  //       // Add the product of the corresponding elements of A and B to C
  //       C[i][j] += A[i][k] * B[k][j];
  //     }
  //   }
  // }
  #pragma omp parallel for
  for(int i = 0; i < n_rows; ++i) {        
    for(int k = 0; k < n_inner; ++k) {       
      for(int j = 0; j < n_cols; ++j) {                
        C[i][j] += A[i][k] * B[k][j];     
      } 
    }    
  }

  // Return the output matrix C
  return C;
}

// Transpose function
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix) {
  // Get the number of rows and columns of the matrix
  int n_rows = matrix.size();
  int n_cols = matrix[0].size();

  // Initialize the transposed matrix as a zero matrix
  std::vector<std::vector<double>> matrix_T(n_cols, std::vector<double>(n_rows, 0.0));

  // Loop over the rows and columns of the matrix
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      // Assign the element of the matrix to the corresponding element of the transposed matrix
      matrix_T[j][i] = matrix[i][j];
    }
  }

  // Return the transposed matrix
  return matrix_T;
}

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
  AX = matmul(A, X);
  // Matrix multiplication of AX and W
  std::vector<std::vector<double>> AXW(AX.size(), std::vector<double>(W[0].size(), 0.0));
  AXW = matmul(AX, W);

//   printf("X--------\n");
//   print_vector_of_vector(X, 10);
//   printf("--------\n");  

//   printf("AX--------\n");
//   print_vector_of_vector(AX, 10);
//   printf("--------\n");  

//   printf("W--------\n");
//   print_vector_of_vector(W, 10);
//   printf("--------\n");  

//   printf("AXW--------\n");
//   print_vector_of_vector(AXW, 10);
//   printf("--------\n"); 

  std::vector<std::vector<double>> Z(AXW.size(), std::vector<double>(AXW[0].size(), 0.0));
  if (layer == 1) {
    // printf("AXW--------\n");
    // print_vector_of_vector(W, 10);
    // printf("--------\n");  
    Z = softmax(AXW);
    // printf("prediction------\n");
    // print_vector_of_vector(AXW, 10);
    // print_vector_of_vector(Z, 10);
    // printf("------\n");
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
std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& W, const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& Z, const std::vector<std::vector<double>>& dZ, double lr, uint64_t layer, size_t n_train) {
  std::vector<std::vector<double>> AX(X.size(), std::vector<double>(X[0].size(), 0.0));
  AX = matmul(A, X); 

  // printf("ah--------\n");
  // print_vector_of_vector(AX);
  // printf("--------\n");

  // printf("p_minus_y--------\n");
  // print_vector_of_vector(dZ);
  // printf("--------\n");

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
  std::vector<std::vector<double>> AX_T = transpose(AX);
  std::vector<std::vector<double>> W_T = transpose(W);
  dW = matmul(AX_T, dAXW);
  dAX = matmul(dAXW, W_T);

  // Gradient of the matrix multiplication of A and X
  std::vector<std::vector<double>> dX(X.size(), std::vector<double>(X[0].size(), 0.0));
  dX = matmul(A, dAX);
  // printf(">>>>>>>>>\n");
  // print_vector_of_vector(W);

  // dW avg across samples
  for (int i = 0; i < dW.size(); i++) {
    for (int j = 0; j < dW[0].size(); j++) {
      dW[i][j] = dW[i][j] / n_train;
    }
  }

  // printf("dW--------\n");
  // print_vector_of_vector(dW, 10);
  // printf("--------\n");
  // printf("W before--------\n");
  // print_vector_of_vector(W);
  // printf("--------\n");
  // Update the weight matrix W with gradient descent
  for (int i = 0; i < W.size(); i++) {
    for (int j = 0; j < W[0].size(); j++) {
      W[i][j] -= lr * dW[i][j];
    }
  }
  // printf("W after--------\n");
  // print_vector_of_vector(W);
  // printf("--------\n");
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
//   printf("Y_pred--------\n");
//   print_vector_of_vector(Y_pred, 10);
//   printf("--------\n");
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

// A function that takes three parameters: Y, Y_pred and is_border
// Y and Y_pred are std::vector<std::vector<double>> representing the labels and predictions
// is_border is a std::vector<bool> indicating which elements are border elements
// The function returns the percentage of correct predictions for border elements
double masked_accuracy(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred, const std::vector<bool>& is_border) {
  // Initialize a counter variable to store the number of correct predictions
  int n_correct = 0;
  // Initialize a counter variable to store the number of border elements
  int n_border = 0;
  // Loop through each element of the vectors
  for (int i = 0; i < Y.size(); i++) {
    // Check if the element is a border element
    if (is_border[i]) {
      // Increment the number of border elements
      n_border++;
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
        // Increment the number of correct predictions
        n_correct++;
      }
    }
  }
  // Return the percentage of correct predictions for border elements
  // If there are no border elements, return 0
  return n_border > 0 ? (double) n_correct / n_border * 100 : 0;
}

void get_split_mask(
  size_t n, 
  double train_ratio, 
  double val_ratio, 
  double test_ratio,
  std::vector<bool>& train_mask,
  std::vector<bool>& val_mask,
  std::vector<bool>& test_mask
) {
  train_mask.resize(n, false);
  val_mask.resize(n, false);
  test_mask.resize(n, false);
  size_t n_train = (size_t)((double)n * train_ratio);
  size_t n_val = (size_t)((double)n * val_ratio);
  size_t n_test = n - n_train - n_val;
  printf("n_train: %lu, n_val: %lu, n_test: %lu\n", n_train, n_val, n_test);
  for (int i = 0; i < n_train; ++i) train_mask[i] = true;
  for (int i = n_train; i < n_train + n_val; ++i) val_mask[i] = true;
  for (int i = n_train + n_val; i < n; ++i) test_mask[i] = true;
}

uint64_t count_true(const std::vector<bool>& vec) {
  // Initialize a counter variable to store the result
  uint64_t count = 0;
  // Loop through each element of the vector
  for (bool b : vec) {
    // If the element is true, increment the counter
    if (b) {
      count++;
    }
  }
  // Return the final count
  return count;
}

std::vector<bool> and_bool_vec(const std::vector<bool>& a, const std::vector<bool>& b) {
  auto c = a;
  for (int i = 0; i < c.size(); ++i) c[i] = (c[i] && b[i]);
  return c;
}

std::vector<bool> or_bool_vec(const std::vector<bool>& a, const std::vector<bool>& b) {
  auto c = a;
  for (int i = 0; i < c.size(); ++i) c[i] = (c[i] || b[i]);
  return c;
}

template <typename T>
std::vector<std::vector<T>> mask_tensor(const std::vector<std::vector<T>>& a, const std::vector<bool>& m) {
  auto ret = a;
  for (int i = 0; i < a.size(); ++i) {
    if (!m[i]) ret[i] = std::vector<T>(ret[i].size(), 0.0);
  }
  return ret;
}

// Symmetric normalization function
std::vector<std::vector<double>> symnorm(std::vector<std::vector<double>> A) {
  // Get the number of nodes
  int n_nodes = A.size();

  for (int i = 0; i < n_nodes; i++) {
    A[i][i] = 1.0;
  }

  // Initialize the degree matrix D as a diagonal matrix
  std::vector<std::vector<double>> D(n_nodes, std::vector<double>(n_nodes, 0.0));
  for (int i = 0; i < n_nodes; i++) {
    D[i][i] = 0.0;
    // Sum the degrees of each node
    for (int j = 0; j < n_nodes; j++) {
      D[i][i] += A[i][j];
    }
    // Take the inverse of the square root of the degrees
    D[i][i] = 1.0 / std::sqrt(D[i][i]);
  }

  // Initialize the weighted adjacency matrix A' as a zero matrix
  std::vector<std::vector<double>> A_prime(n_nodes, std::vector<double>(n_nodes, 0.0));

  // Compute the symmetric normalization formula
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      A_prime[i][j] = D[i][i] * A[i][j] * D[j][j];
    }
  }

  // Return the weighted adjacency matrix A'
  return A_prime;
}

// Read X, Y from a vertex file and A from an edge file function
void read_data(std::string vertex_file, std::string edge_file, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& Y, std::vector<std::vector<double>>& A) {
  // Open the vertex file
  std::ifstream vfile(vertex_file);
  if (!vfile.is_open()) {
    std::cerr << "Error: cannot open the vertex file" << std::endl;
    return;
  }

  // Initialize the vertex feature matrix X and the label matrix Y as zero matrices
  X = std::vector<std::vector<double>>(n_nodes, std::vector<double>(n_features, 0.0));
  Y = std::vector<std::vector<double>>(n_nodes, std::vector<double>(n_classes, 0.0));

  // Read the vertex file line by line
  std::string line;
  while (std::getline(vfile, line)) {
    // Split the line by whitespace
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, ' ')) {
      tokens.push_back(token);
    }
    // Check if the line has the correct format
    if (tokens.size() != n_features + 2) {
      std::cerr << "Error: invalid vertex file format" << std::endl;
      return;
    }
    // Get the sample index, the features, and the label from the line
    int index = std::stoi(tokens[0]);
    std::vector<double> features(n_features, 0.0);
    std::vector<double> label(n_classes, 0.0);
    for (int i = 0; i < n_features; i++) {
      features[i] = std::stod(tokens[i + 1]);
    }
    int class_index = std::stoi(tokens[n_features + 1]);
    label[class_index] = 1.0;
    // Assign the features and the label to the corresponding row of X and Y
    X[index] = features;
    Y[index] = label;
  }

  // Close the vertex file
  vfile.close();

  // Open the edge file
  std::ifstream efile(edge_file);
  if (!efile.is_open()) {
    std::cerr << "Error: cannot open the edge file" << std::endl;
    return;
  }

  // Initialize the adjacency matrix A as a zero matrix
  A = std::vector<std::vector<double>>(n_nodes, std::vector<double>(n_nodes, 0.0));

  // Read the edge file line by line
  while (std::getline(efile, line)) {
    // Split the line by whitespace
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, ' ')) {
      tokens.push_back(token);
    }
    // Check if the line has the correct format
    if (tokens.size() != 2) {
      std::cerr << "Error: invalid edge file format" << std::endl;
      return;
    }
    // Get the indices of the two vertices from the line
    int i = std::stoi(tokens[0]);
    int j = std::stoi(tokens[1]);
    // Assign 1 to the corresponding element of A
    A[i][j] = 1.0;
    A[j][i] = 1.0; // Assuming the graph is undirected
  }

  // Close the edge file
  efile.close();

  // Return nothing
  return;
}

// A function that takes a file name as input and returns a vector of partitions
std::vector<int> read_partition_file(std::string file_name) {
  // Create an empty vector to store the partitions
  std::vector<int> partitions;
  // Open the file for reading
  std::ifstream fin(file_name);
  // Check if the file is opened successfully
  if (fin.is_open()) {
    // Declare variables to store the vertex index and partition
    int vertex, partition;
    // Read the file line by line
    while (fin >> vertex >> partition) {
      // Push the partition to the vector
      partitions.push_back(partition);
    }
    // Close the file
    fin.close();
  }
  else {
    // Print an error message if the file cannot be opened
    std::cout << "Error: Cannot open the file " << file_name << std::endl;
  }
  // Return the vector of partitions
  return partitions;
}

// GCN weight initialization function using the Glorot method
std::vector<std::vector<double>> init_weight(int n_features, int n_classes) {
  // Initialize the weight matrix W as a zero matrix
  std::vector<std::vector<double>> W(n_features, std::vector<double>(n_classes, 0.0));

  // Set the random seed
  std::srand(42);

  // Compute the Glorot uniform limit
  double limit = std::sqrt(6.0 / (n_features + n_classes));

  // Loop over the rows and columns of the weight matrix
  for (int i = 0; i < n_features; i++) {
    for (int j = 0; j < n_classes; j++) {
      // Assign a random value between -limit and limit to the weight matrix element
      W[i][j] = (double) std::rand() / RAND_MAX * 2 * limit - limit;
    }
  }

  // Return the weight matrix W
  return W;
}

// Feature normalization function using the row normalization
std::vector<std::vector<double>> normalize_feature(std::vector<std::vector<double>> X) {
  // Get the number of nodes and features
  int n_nodes = X.size();
  int n_features = X[0].size();

  // Initialize the normalized feature matrix X' as a zero matrix
  std::vector<std::vector<double>> X_prime(n_nodes, std::vector<double>(n_features, 0.0));

  // Loop over the nodes
  for (int i = 0; i < n_nodes; i++) {
    // Compute the sum of the features of the node
    double sum = 0.0;
    for (int j = 0; j < n_features; j++) {
      sum += X[i][j];
    }
    if (sum == 0.0) sum = 1.0;
    // Normalize the features by dividing by the sum
    for (int j = 0; j < n_features; j++) {
      X_prime[i][j] = X[i][j] / sum;
    }
  }

  // Return the normalized feature matrix X'
  return X_prime;
}

// A function that takes four vectors as parameters and returns a vector of tuples containing new A, Y, X for each partition
std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<bool>>> generate_new_vectors(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& Y, 
    const std::vector<std::vector<double>>& X, 
    const std::vector<int>& partition
) {
  // A vector to store the results
  std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<bool>>> results;
  // A map to store the nodes in each partition
  std::map<int, std::vector<int>> partitions;
  // Loop through the partition vector and group the nodes by their partition number
  for (int i = 0; i < partition.size(); i++) {
    partitions[partition[i]].push_back(i);
  }
  // Loop through the partitions map and generate new A, Y, X for each partition
  for (auto& p : partitions) {
    std::cout << "Loop to partition " << p.first << std::endl;
    // Get the partition number and the nodes in that partition
    int p_num = p.first;
    std::vector<int> p_nodes = p.second;
    // Get the number of nodes, classes, and features in that partition
    int n_nodes = p_nodes.size();
    int n_classes = Y[0].size();
    int n_features = X[0].size();
    // Initialize new A, Y, X with zeros
    std::vector<std::vector<double>> new_A(n_nodes, std::vector<double>(n_nodes, 0.0));
    std::vector<std::vector<double>> new_Y(n_nodes, std::vector<double>(n_classes, 0.0));
    std::vector<std::vector<double>> new_X(n_nodes, std::vector<double>(n_features, 0.0));
    std::vector<bool> new_is_border(n_nodes, false);
    // Loop through the nodes in the partition and copy the corresponding values from the original vectors
    for (int i = 0; i < n_nodes; i++) {
      for (int j = 0; j < A[0].size(); j++) {
        if (partition[j] != p_num && A[p_nodes[i]][j] != 0.0) {
            new_is_border[i] = true;
        } 
      }
      for (int j = 0; j < n_nodes; j++) {
        new_A[i][j] = A[p_nodes[i]][p_nodes[j]];
      }
      for (int k = 0; k < n_classes; k++) {
        new_Y[i][k] = Y[p_nodes[i]][k];
      }
      for (int l = 0; l < n_features; l++) {
        new_X[i][l] = X[p_nodes[i]][l];
      }
    }
    // Create a tuple containing new A, Y, X and push it to the results vector
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<bool>> new_tuple(new_A, new_Y, new_X, new_is_border);
    results.push_back(new_tuple);
  }
  // Return the results vector
  return results;
}

size_t count_inter_partition_edges(const std::vector<std::vector<double>>& A, const std::vector<int>& partition) {
    size_t count = 0;
    // Iterate over each edge in the adjacency matrix
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            // Check if there is an edge and the vertices belong to different partitions
            if (A[i][j] != 0.0 && partition[i] != partition[j]) {
                ++count;
            }
        }
    }
    // Since the adjacency matrix is symmetric and we counted each edge twice, divide by 2
    return count;
}

std::vector<std::vector<double>> mat_scale(const std::vector<std::vector<double>>& mat, double scaler) {
    std::vector<std::vector<double>> ret = mat;
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat[0].size(); ++j) {
            ret[i][j] *= scaler;
        }
    }
    return ret;
}

std::vector<std::vector<double>> mat_add(const std::vector<std::vector<double>>& mat0, const std::vector<std::vector<double>>& mat1) {
    std::vector<std::vector<double>> ret = mat0;
    for (int i = 0; i < mat0.size(); ++i) {
        for (int j = 0; j < mat0[0].size(); ++j) {
            ret[i][j] += mat1[i][j];
        }
    }
    return ret;
}

// Define the main function
int main(int argc, char* argv[]) {
	n_parts = std::atoi(argv[1]);
  p_path = std::string("/home/zzh/project/SecGNN/data/") + Dataset + "/transformed/" + dataset + ".part.preprocessed." + std::to_string(n_parts) + std::string("p");

  // Set the random seed
  std::srand(42);

  // Define the weighted adjacency matrix A (symmetric and normalized)
  std::vector<std::vector<double>> A(n_nodes, std::vector<double>(n_nodes, 0.0));
  std::vector<std::vector<double>> Y(n_nodes, std::vector<double>(n_classes, 0.0));
  std::vector<std::vector<double>> X(n_nodes, std::vector<double>(n_features, 0.0));

  read_data(v_path, e_path, X, Y, A);
  std::vector<int> partition = read_partition_file(p_path);
//   std::map<int, std::vector<std::vector<double>>> parted_As = partition_adjacency_matrix(partition, A);
//   for (int i = 0 ; i < n_parts; ++i) {
//     parted_As[i] = symnorm(parted_As[i]);
//   }

  std::vector<std::vector<double>> W1_init(n_features, std::vector<double>(n_hidden, 0.0));
  W1_init = init_weight(n_features, n_hidden);
  std::vector<std::vector<double>> W2_init(n_hidden, std::vector<double>(n_classes, 0.0));
  W2_init = init_weight(n_hidden, n_classes);

  auto parted = generate_new_vectors(A, Y, X, partition);
  auto num_inters = count_inter_partition_edges(A, partition);
  printf(">>>>> num of inter-partition edges is %lu\n", num_inters);
  std::vector<std::vector<std::vector<double>>> As;
  std::vector<std::vector<std::vector<double>>> Ys;
  std::vector<std::vector<std::vector<double>>> Xs;
  std::vector<std::vector<bool>> is_borders;
  std::vector<std::vector<std::vector<double>>> W1s;
  std::vector<std::vector<std::vector<double>>> W2s;
  std::vector<std::vector<bool>> train_masks(n_parts), val_masks(n_parts), test_masks(n_parts);

  // A = symnorm(A);
  for (int i = 0; i < n_parts; ++i) {
    As.push_back(std::get<0>(parted[i]));
    As[i] = symnorm(As[i]);
    Ys.push_back(std::get<1>(parted[i]));
    Xs.push_back(std::get<2>(parted[i]));
    // Xs[i] = normalize_feature(Xs[i]);
    is_borders.push_back(std::get<3>(parted[i]));

    printf("num of is_borders %lu of part %d\n", count_true(is_borders[i]), i);

    W1s.push_back(W1_init);
    W2s.push_back(W2_init);

    get_split_mask(As[i].size(), train_ratio, val_ratio, test_ratio, train_masks[i], val_masks[i], test_masks[i]);
  }
  // A = symnorm(A);

  // X = normalize_feature(X);

//   // Define the weight matrices W1 and W2 (randomly initialized)
//   std::vector<std::vector<double>> W1_init(n_features, std::vector<double>(n_hidden, 0.0));
//   W1_init = init_weight(n_features, n_hidden);
//   // for (int i = 0; i < n_features; i++) {
//   //   for (int j = 0; j < n_hidden; j++) {
//   //     W1[i][j] = (double) std::rand() / RAND_MAX;
//   //     // W1[i][j] = 0.5;
//   //   }
//   // }
//   std::vector<std::vector<double>> W2_init(n_hidden, std::vector<double>(n_classes, 0.0));
//   W2_init = init_weight(n_hidden, n_classes);
//   // for (int i = 0; i < n_hidden; i++) {
//   //   for (int j = 0; j < n_classes; j++) {
//   //     W2[i][j] = (double) std::rand() / RAND_MAX;
//   //     // W2[i][j] = 0.05;
//   //   }
//   // }

  // Train the two-layer GCN
  for (int epoch = 0; epoch < n_epochs; epoch++) {

    for (int p = 0; p < n_parts; ++p) {
        std::vector<std::vector<double>>& W1 = W1s[p];
        std::vector<std::vector<double>>& W2 = W2s[p];
        std::vector<std::vector<double>>& A = As[p];
        std::vector<std::vector<double>>& Y = Ys[p];
        std::vector<std::vector<double>>& X = Xs[p];
        std::vector<bool>& is_border = is_borders[p];
        size_t part_n_nodes = A.size();
        std::vector<bool>& train_mask = train_masks[p];
        std::vector<bool>& val_mask = val_masks[p];
        std::vector<bool>& test_mask = test_masks[p];

        // Forward pass
        // printf("H1\n");
        std::vector<std::vector<double>> Z1 = forward(X, W1, A, 0); // First layer output
        // printf("H2\n");
        std::vector<std::vector<double>> Z2 = forward(Z1, W2, A, 1); // Second layer output
        std::vector<std::vector<double>> Y_pred = Z2; // Final prediction
        // printf("H3\n");
        // Compute the loss
        double los = cross_entropy_loss(Y, Y_pred);
        // printf("H4\n");
        // Compute the accuracy
        double acc = accuracy(Y, Y_pred);
        double train_acc = masked_accuracy(Y, Y_pred, train_mask);
        double val_acc = 0.0;
        if (val_ratio > 0.0) val_acc = masked_accuracy(Y, Y_pred, val_mask);
        double test_acc = 0.0;
        double test_is_border_acc = 0.0;
        if (test_ratio > 0.0) {
          test_acc = masked_accuracy(Y, Y_pred, test_mask);
          test_is_border_acc = masked_accuracy(Y, Y_pred, and_bool_vec(test_mask, is_border));
        }
        // printf("H5\n");

        // Print the loss and the accuracy
        std::cout << "Epoch " << epoch + 1 << ", Part: " << p << ", Loss: " << los 
          << ", Full accuracy: " << acc << "%" 
          << ", train accuracy: " << train_acc << "%" 
          << ", Val accuracy: " << val_acc << "%" 
          << ", Test accuracy: " << test_acc << "%"
          << ", Test is_border accuracy: " << test_is_border_acc << "%" << std::endl;
        // printf("H5\n");

        // Print the loss and the accuracy
        // std::cout << "Epoch " << epoch + 1 << ", Part: " << p << ", Loss: " << los << ", Accuracy: " << acc << "%" << std::endl;

        // Backward pass
        std::vector<std::vector<double>> dY_pred = dloss(Y, Y_pred); // Gradient of the loss
        dY_pred = mask_tensor(dY_pred, train_mask);
        size_t part_n_train = part_n_nodes * train_ratio;
        auto tmp = backward(Z1, W2, A, Z2, dY_pred, lr, 1, part_n_train); // Update W2
        // printf("H6\n");
        std::vector<std::vector<double>> dZ1 = tmp; // Gradient of the first layer output
        backward(X, W1, A, Z1, dZ1, lr, 0, part_n_train); // Update W1
        // printf("H7\n");
    }

    double scaler = (double) 1 / n_parts;
    for (int p = 1; p < n_parts; p++) {
        W1s[0] = mat_add(W1s[0], W1s[p]);
        W2s[0] = mat_add(W2s[0], W2s[p]);
    }
    W1s[0] = mat_scale(W1s[0], scaler);
    W2s[0] = mat_scale(W2s[0], scaler);
    for (int p = 1; p < n_parts; p++) {
        W1s[p] = W1s[0];
        W2s[p] = W2s[0];
    }    
  }

  // Return 0
  return 0;
}