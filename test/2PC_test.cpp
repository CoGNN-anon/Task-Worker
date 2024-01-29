#include "TaskqHandler.h"
#include "TaskUtil.h"
#include "SCIHarness.h"
#include "task.h"
#include "troy/app/TroyFHEWrapper.cuh"

#include <vector>

using namespace troyn;


const uint64_t DataBound = (1ul << 44);
const uint64_t Mod = (1ul << 44);
const uint64_t Scaler = (1ul << 8);

void test_twoPartyGCNVectorScale(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNVectorScale\n");

    size_t taskNum = 2;
    std::vector<std::vector<double>> embeddingVecs_plain = {
        // {10, 12, 4, 5, 2},
        // {1, 2, 3, 4, 5}
        {0.0898438, 0.90625}, 
        {0.617188, 0.378906}, 
    };
    std::vector<std::vector<double>> expected = embeddingVecs_plain;
    std::vector<double> scaler0_plain = {
        // 2,
        // 3
        0.574219,
        0.574219
    };
    std::vector<double> scaler1_plain = {
        // 0.3,
        // 0.1
        0.574219,
        0.574219
    };

    for (int i = 0; i < embeddingVecs_plain.size(); ++i) {
        for (int j = 0; j < embeddingVecs_plain[i].size(); ++j) expected[i][j] = embeddingVecs_plain[i][j] * scaler0_plain[i] * scaler1_plain[i];
    } 

    std::vector<std::vector<uint64_t>> embeddingVecs(taskNum);
    std::vector<uint64_t> scaler0(taskNum);
    std::vector<uint64_t> scaler1(taskNum);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embeddingVecs1;
        std::vector<uint64_t> scaler01;
        std::vector<uint64_t> scaler11;
        CryptoUtil::intoShares(embeddingVecs_plain, embeddingVecs, embeddingVecs1);
        CryptoUtil::intoShares(scaler0_plain, scaler0, scaler01);
        CryptoUtil::intoShares(scaler1_plain, scaler1, scaler11);
        clientTaskComm.sendShareVecVec(embeddingVecs1, 1 - partyId);
        clientTaskComm.sendShareVec(scaler01, 1 - partyId);
        clientTaskComm.sendShareVec(scaler11, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(embeddingVecs, 1 - partyId);
        serverTaskComm.recvShareVec(scaler0, 1 - partyId);
        serverTaskComm.recvShareVec(scaler1, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> result;
    sci::twoPartyGCNVectorScale(embeddingVecs, scaler0, scaler1, result, 1-partyId, role);
    result.swap(embeddingVecs);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embeddingVecs1;
        clientTaskComm.recvShareVecVec(embeddingVecs1, 1 - partyId);
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(embeddingVecs_plain, embeddingVecs, embeddingVecs1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected);
        printf("Actual:\n");
        sci::print_vector_of_vector(embeddingVecs_plain);
        printf("Test print:\n");
        sci::printShareVecVec(embeddingVecs, 1 - partyId, sci::ALICE);
    } else {
        serverTaskComm.sendShareVecVec(embeddingVecs, 1 - partyId);
        sci::printShareVecVec(embeddingVecs, 1 - partyId, sci::BOB);
    }
}

void test_twoPartyGCNSingleVectorScale(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNSingleVectorScale\n");

    size_t taskNum = 2;
    std::vector<std::vector<double>> embeddingVecs_plain = {
        // {10, 12, 4, 5, 2},
        // {1, 2, 3, 4, 5}
        {0.0898438, 0.90625}, 
        {0.617188, 0.378906}, 
    };
    std::vector<std::vector<double>> expected = embeddingVecs_plain;
    std::vector<double> scaler0_plain = {
        // 2,
        // 3
        0.574219,
        0.574219
    };

    for (int i = 0; i < embeddingVecs_plain.size(); ++i) {
        for (int j = 0; j < embeddingVecs_plain[i].size(); ++j) expected[i][j] = embeddingVecs_plain[i][j] * scaler0_plain[i];
    } 

    std::vector<std::vector<uint64_t>> embeddingVecs(taskNum);
    std::vector<uint64_t> scaler0(taskNum);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embeddingVecs1;
        std::vector<uint64_t> scaler01(taskNum);
        CryptoUtil::intoShares(embeddingVecs_plain, embeddingVecs, embeddingVecs1);
        // CryptoUtil::intoShares(scaler0_plain, scaler0, scaler01);
        for (int i = 0; i < taskNum; ++i) {
            scaler0[i] = (uint64_t)(scaler0_plain[i] * (1 << SCALER_BIT_LENGTH)); 
            scaler01[i] = 0;
        }
        clientTaskComm.sendShareVecVec(embeddingVecs1, 1 - partyId);
        clientTaskComm.sendShareVec(scaler01, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(embeddingVecs, 1 - partyId);
        serverTaskComm.recvShareVec(scaler0, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> result;
    sci::twoPartyGCNVectorScale(embeddingVecs, scaler0, result, false, 1-partyId, role);
    result.swap(embeddingVecs);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embeddingVecs1;
        clientTaskComm.recvShareVecVec(embeddingVecs1, 1 - partyId);
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(embeddingVecs_plain, embeddingVecs, embeddingVecs1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected);
        printf("Actual:\n");
        sci::print_vector_of_vector(embeddingVecs_plain);
        printf("Test print:\n");
        sci::printShareVecVec(embeddingVecs, 1 - partyId, sci::ALICE);
    } else {
        serverTaskComm.sendShareVecVec(embeddingVecs, 1 - partyId);
        sci::printShareVecVec(embeddingVecs, 1 - partyId, sci::BOB);
    }
}

void test_twoPartyGCNCondVectorAddition(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNCondVectorAddition\n");

    size_t taskNum = 3;
    std::vector<std::vector<double>> operands0_plain = {
        {10, 12, 4, 5, 2},
        {1, 2, 3, 4, 5},
        {4, 9, 1, 1, 0}
    };
    std::vector<std::vector<double>> operands1_plain = {
        {7, 3, 2, 6, 9},
        {1.9, 2.5, 1.3, 1.4, 0.5},
        {4, 9.9, 1.0, 1, 0}
    };
    std::vector<bool> cond_plain = {
        true,
        true,
        true
    };
    std::vector<std::vector<double>> expected = operands0_plain;    

    for (int i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < expected[i].size(); ++j) expected[i][j] = operands0_plain[i][j] + operands1_plain[i][j];
    } 

    std::vector<std::vector<uint64_t>> operands0(taskNum);
    std::vector<std::vector<uint64_t>> operands1(taskNum);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> operands01(taskNum);
        std::vector<std::vector<uint64_t>> operands11(taskNum);
        CryptoUtil::intoShares(operands0_plain, operands0, operands01);
        CryptoUtil::intoShares(operands1_plain, operands1, operands11);
        clientTaskComm.sendShareVecVec(operands01, 1 - partyId);
        clientTaskComm.sendShareVecVec(operands11, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(operands0, 1 - partyId);
        serverTaskComm.recvShareVecVec(operands1, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> result;
    sci::twoPartyGCNCondVectorAddition(operands0, operands1, cond_plain, result, 1-partyId, role);
    result.swap(operands0);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> operands01;
        clientTaskComm.recvShareVecVec(operands01, 1 - partyId);
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(operands0_plain, operands0, operands01);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected);
        printf("Actual:\n");
        sci::print_vector_of_vector(operands0_plain);
    } else {
        serverTaskComm.sendShareVecVec(operands0, 1 - partyId);
    }
}

void matrix_multiply_relu(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C, std::vector<std::vector<double>>& D) {
  // Get the dimensions of A and B
  int m = A.size(); // Number of rows in A
  int n = A[0].size(); // Number of columns in A
  int p = B[0].size(); // Number of columns in B
  // Initialize the result matrix C with zeros
  C.resize(m, std::vector<double>(p, 0));
  D.resize(m, std::vector<double>(p, 0));
  // Loop through the rows of A
  for (int i = 0; i < m; i++) {
    // Loop through the columns of B
    for (int j = 0; j < p; j++) {
      // Loop through the columns of A
      for (int k = 0; k < n; k++) {
        // Multiply the corresponding elements of A and B and add to C[i][j]
        C[i][j] += A[i][k] * B[k][j];
      }
      // Apply Relu to C[i][j] by setting it to zero if it is negative
      D[i][j] = std::max(0.0, C[i][j]);
    }
  }
}

void test_twoPartyGCNForwardNN(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNForwardNN\n");

    // const ShareVecVec& _embedding, 
    // const ShareVecVec& _weight, 
    // ShareVecVec& _z, 
    // ShareVecVec& _new_h, 

    size_t taskNum = 3;
    std::vector<std::vector<double>> embedding_plain = {
        {10, 12, 4, 5, 2},
        {1, 2, 3, 4, 5},
        {4, 9, 1, 1, 0}
    };
    std::vector<uint64_t> normalizer(taskNum, 1 << SCALER_BIT_LENGTH); 
    std::vector<uint64_t> zero_normalizer(taskNum, 0); 
    std::vector<std::vector<double>> weight_plain = {
        {7, 3},
        {-1, 2.5},
        {4, -9.9},
        {1.3, 1.4},
        {1.0, 1}
    };
    std::vector<std::vector<double>> expected_z;
    std::vector<std::vector<double>> expected_new_h;
    matrix_multiply_relu(embedding_plain, weight_plain, expected_z, expected_new_h);

    std::vector<std::vector<uint64_t>> embedding;
    std::vector<std::vector<uint64_t>> weight;

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embedding1;
        std::vector<std::vector<uint64_t>> weight1;
        CryptoUtil::intoShares(embedding_plain, embedding, embedding1);
        CryptoUtil::intoShares(weight_plain, weight, weight1);
        clientTaskComm.sendShareVecVec(embedding1, 1 - partyId);
        clientTaskComm.sendShareVecVec(weight1, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(embedding, 1 - partyId);
        serverTaskComm.recvShareVecVec(weight, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> z;
    std::vector<std::vector<uint64_t>> new_h;
    if (role == sci::ALICE) sci::twoPartyGCNForwardNN(embedding, weight, normalizer, z, new_h, 1-partyId, role);
    else sci::twoPartyGCNForwardNN(embedding, weight, zero_normalizer, z, new_h, 1-partyId, role);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> z1;
        std::vector<std::vector<uint64_t>> new_h1;
        clientTaskComm.recvShareVecVec(z1, 1 - partyId);
        clientTaskComm.recvShareVecVec(new_h1, 1 - partyId);
        std::vector<std::vector<double>> z_plain;
        std::vector<std::vector<double>> new_h_plain;
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(z_plain, z, z1);
        CryptoUtil::mergeShareAsDouble(new_h_plain, new_h, new_h1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected_z);
        sci::print_vector_of_vector(expected_new_h);
        printf("Actual:\n");
        sci::print_vector_of_vector(z_plain);
        sci::print_vector_of_vector(new_h_plain);
    } else {
        serverTaskComm.sendShareVecVec(z, 1 - partyId);
        serverTaskComm.sendShareVecVec(new_h, 1 - partyId);
    }
}

void matrix_multiply_softmax_minus(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, const std::vector<std::vector<double>>& E, std::vector<std::vector<double>>& C, std::vector<std::vector<double>>& D, std::vector<std::vector<double>>& F) {
  // Get the dimensions of A and B
  int m = A.size(); // Number of rows in A
  int n = A[0].size(); // Number of columns in A
  int p = B[0].size(); // Number of columns in B
  // Initialize the result matrix C with zeros
  C.resize(m, std::vector<double>(p, 0));
  D.resize(m, std::vector<double>(p, 0));
  F.resize(m, std::vector<double>(p, 0));
  // Loop through the rows of A
  for (int i = 0; i < m; i++) {
    // Loop through the columns of B
    for (int j = 0; j < p; j++) {
      // Loop through the columns of A
      for (int k = 0; k < n; k++) {
        // Multiply the corresponding elements of A and B and add to C[i][j]
        C[i][j] += A[i][k] * B[k][j];
      }
      // Apply Softmax to C[i][j] by exponentiating it and dividing by the sum of the row
      double sum = 0; // The sum of the row
      for (int l = 0; l < p; l++) {
        sum += std::exp(C[i][l]); // Add the exponentiated element to the sum
      }
      D[i][j] = std::exp(C[i][j]) / sum; // Divide the exponentiated element by the sum
      F[i][j] = D[i][j] - E[i][j];
    }
  }
}

void test_twoPartyGCNForwardNNPrediction(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNForwardNNPrediction\n");

    // const ShareVecVec& _embedding, 
    // const ShareVecVec& _weight, 
    // const ShareVecVec& _y,
    // ShareVecVec& _z, 
    // ShareVecVec& _p, 
    // ShareVecVec& _p_minus_y, 

    size_t taskNum = 3;
    std::vector<std::vector<double>> embedding_plain = {
        {1, 1.2, 0.4, 0.5, 0.2},
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {0.4, 0.9, 0.1, 0.1, 0}
    };
    std::vector<std::vector<double>> weight_plain = {
        {7, 3},
        {-1, 2.5},
        {4, -9.9},
        {1.3, 1.4},
        {1.0, 1}
    };
    std::vector<std::vector<double>> y_plain = {
        {0, 1},
        {1, 0},
        {2, 0}
    };
    std::vector<uint64_t> normalizer(taskNum, 1 << SCALER_BIT_LENGTH); 
    std::vector<uint64_t> zero_normalizer(taskNum, 0); 

    std::vector<std::vector<double>> expected_z;
    std::vector<std::vector<double>> expected_p;
    std::vector<std::vector<double>> expected_p_minus_y;
    matrix_multiply_softmax_minus(embedding_plain, weight_plain, y_plain, expected_z, expected_p, expected_p_minus_y);

    std::vector<std::vector<uint64_t>> embedding;
    std::vector<std::vector<uint64_t>> weight;
    std::vector<std::vector<uint64_t>> y;

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> embedding1;
        std::vector<std::vector<uint64_t>> weight1;
        std::vector<std::vector<uint64_t>> y1;
        CryptoUtil::intoShares(embedding_plain, embedding, embedding1);
        CryptoUtil::intoShares(weight_plain, weight, weight1);
        CryptoUtil::intoShares(y_plain, y, y1);
        clientTaskComm.sendShareVecVec(embedding1, 1 - partyId);
        clientTaskComm.sendShareVecVec(weight1, 1 - partyId);
        clientTaskComm.sendShareVecVec(y1, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(embedding, 1 - partyId);
        serverTaskComm.recvShareVecVec(weight, 1 - partyId);
        serverTaskComm.recvShareVecVec(y, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> z;
    std::vector<std::vector<uint64_t>> p;
    std::vector<std::vector<uint64_t>> p_minus_y;
    if (role == sci::ALICE) sci::twoPartyGCNForwardNNPrediction(embedding, weight, y, normalizer, z, p, p_minus_y, 1-partyId, role);
    else sci::twoPartyGCNForwardNNPrediction(embedding, weight, y, zero_normalizer, z, p, p_minus_y, 1-partyId, role);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> z1;
        std::vector<std::vector<uint64_t>> p1;
        std::vector<std::vector<uint64_t>> p_minus_y1;
        clientTaskComm.recvShareVecVec(z1, 1 - partyId);
        clientTaskComm.recvShareVecVec(p1, 1 - partyId);
        clientTaskComm.recvShareVecVec(p_minus_y1, 1 - partyId);
        std::vector<std::vector<double>> z_plain;
        std::vector<std::vector<double>> p_plain;
        std::vector<std::vector<double>> p_minus_y_plain;
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(z_plain, z, z1);
        CryptoUtil::mergeShareAsDouble(p_plain, p, p1);
        CryptoUtil::mergeShareAsDouble(p_minus_y_plain, p_minus_y, p_minus_y1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected_z);
        sci::print_vector_of_vector(expected_p);
        sci::print_vector_of_vector(expected_p_minus_y);
        printf("Actual:\n");
        sci::print_vector_of_vector(z_plain);
        sci::print_vector_of_vector(p_plain);
        sci::print_vector_of_vector(p_minus_y_plain);
    } else {
        serverTaskComm.sendShareVecVec(z, 1 - partyId);
        serverTaskComm.sendShareVecVec(p, 1 - partyId);
        serverTaskComm.sendShareVecVec(p_minus_y, 1 - partyId);
    }
}

void matrix_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  // Get the dimensions of A and B
  int m = A.size(); // Number of rows in A
  int n = A[0].size(); // Number of columns in A
  int p = B[0].size(); // Number of columns in B
  // Initialize the result matrix C with zeros
  C.resize(m, std::vector<double>(p, 0));
  // Loop through the rows of A
  for (int i = 0; i < m; i++) {
    // Loop through the columns of B
    for (int j = 0; j < p; j++) {
      // Loop through the columns of A
      for (int k = 0; k < n; k++) {
        // Multiply the corresponding elements of A and B and add to C[i][j]
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test_twoPartyGCNBackwardNNInit(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNBackwardNNInit\n");

    // const ShareVecVec& _p_minus_y,
    // const ShareVecVec& _ah_t,
    // const ShareVecVec& _weight_t,
    // ShareVecVec& _d,
    // ShareVecVec& _g,

    size_t taskNum = 3;
    std::vector<std::vector<double>> p_minus_y_plain = {
        {0.9, 0.8},
        {0.5, 0},
        {1, 0.2}
    };
    std::vector<std::vector<double>> ah_t_plain = {
        {7, 3, 1},
        {-1, 2.5, 4},
        {0.1, 0.2, 0.1}
    };
    std::vector<std::vector<double>> weight_t_plain = {
        {2.5, 3.3, 0.4},
        {-4, 0.5, 0.4}
    };    
    std::vector<uint64_t> normalizer(taskNum, 1 << SCALER_BIT_LENGTH); 
    std::vector<uint64_t> zero_normalizer(taskNum, 0); 
    if (role == sci::ALICE) normalizer = zero_normalizer;

    std::vector<std::vector<double>> expected_d;
    std::vector<std::vector<double>> expected_g;
    matrix_multiply(ah_t_plain, p_minus_y_plain, expected_d);
    matrix_multiply(p_minus_y_plain, weight_t_plain, expected_g);

    std::vector<std::vector<uint64_t>> p_minus_y;
    std::vector<std::vector<uint64_t>> ah_t;
    std::vector<std::vector<uint64_t>> weight_t;

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> p_minus_y1;
        std::vector<std::vector<uint64_t>> ah_t1;
        std::vector<std::vector<uint64_t>> weight_t1;
        CryptoUtil::intoShares(p_minus_y_plain, p_minus_y, p_minus_y1);
        CryptoUtil::intoShares(ah_t_plain, ah_t, ah_t1);
        CryptoUtil::intoShares(weight_t_plain, weight_t, weight_t1);
        clientTaskComm.sendShareVecVec(p_minus_y1, 1 - partyId);
        clientTaskComm.sendShareVecVec(ah_t1, 1 - partyId);
        clientTaskComm.sendShareVecVec(weight_t1, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(p_minus_y, 1 - partyId);
        serverTaskComm.recvShareVecVec(ah_t, 1 - partyId);
        serverTaskComm.recvShareVecVec(weight_t, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> d;
    std::vector<std::vector<uint64_t>> g;
    sci::twoPartyGCNBackwardNNInit(p_minus_y, ah_t, weight_t, normalizer, d, g, 1-partyId, role);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> d1;
        std::vector<std::vector<uint64_t>> g1;
        clientTaskComm.recvShareVecVec(d1, 1 - partyId);
        clientTaskComm.recvShareVecVec(g1, 1 - partyId);
        std::vector<std::vector<double>> d_plain;
        std::vector<std::vector<double>> g_plain;
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(d_plain, d, d1);
        CryptoUtil::mergeShareAsDouble(g_plain, g, g1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected_d);
        sci::print_vector_of_vector(expected_g);
        printf("Actual:\n");
        sci::print_vector_of_vector(d_plain);
        sci::print_vector_of_vector(g_plain);
    } else {
        serverTaskComm.sendShareVecVec(d, 1 - partyId);
        serverTaskComm.sendShareVecVec(g, 1 - partyId);
    }
}

void elementwise_matrix_drelu_multiply(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, std::vector<std::vector<double>>& C) {
  // Get the dimensions of A and B
  int m = A.size(); // Number of rows in A
  int n = A[0].size(); // Number of columns in A
  // Initialize the result matrix C with zeros
  C.resize(m, std::vector<double>(n, 0));
  // Loop through the rows of A
  for (int i = 0; i < m; i++) {
    // Loop through the columns of B
    for (int j = 0; j < n; j++) {
        double tmp = A[i][j] > 0.0 ? 1.0 : 0.0;
        C[i][j] = tmp * B[i][j];
    }
  }
}

void test_twoPartyGCNBackwardNN(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyGCNBackwardNN\n");

    // const ShareVecVec& _a_t_g,
    // const ShareVecVec& _ah_t,
    // const ShareVecVec& _z,
    // const ShareVecVec& _weight_t,
    // ShareVecVec& _d,
    // ShareVecVec& _g,
    // int dstTid,

    size_t taskNum = 3;
    std::vector<std::vector<double>> a_t_g_plain = {
        {0.9, 0.8},
        {0.5, 0},
        {1, 0.2}
    };
    std::vector<std::vector<double>> ah_t_plain = {
        {7, 3, 1},
        {-1, 2.5, 4},
        {0.1, 0.2, 0.1}
    };
    std::vector<std::vector<double>> z_plain = {
        {2.5, 3.3},
        {-0.1, 0.5},
        {0.1, 0}
    }; 
    std::vector<std::vector<double>> weight_t_plain = {
        {2.5, 3.3, 0.4},
        {-4, 0.5, 0.4}
    };    
    std::vector<uint64_t> normalizer(taskNum, 1 << SCALER_BIT_LENGTH); 
    std::vector<uint64_t> zero_normalizer(taskNum, 0); 
    if (role == sci::ALICE) normalizer = zero_normalizer;
    
    std::vector<std::vector<double>> expected_d;
    std::vector<std::vector<double>> expected_g;
    std::vector<std::vector<double>> dz_a_t_g;
    elementwise_matrix_drelu_multiply(z_plain, a_t_g_plain, dz_a_t_g);
    matrix_multiply(ah_t_plain, dz_a_t_g, expected_d);
    matrix_multiply(dz_a_t_g, weight_t_plain, expected_g);

    std::vector<std::vector<uint64_t>> a_t_g;
    std::vector<std::vector<uint64_t>> ah_t;
    std::vector<std::vector<uint64_t>> z;
    std::vector<std::vector<uint64_t>> weight_t;

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> a_t_g1;
        std::vector<std::vector<uint64_t>> ah_t1;
        std::vector<std::vector<uint64_t>> z1;
        std::vector<std::vector<uint64_t>> weight_t1;
        CryptoUtil::intoShares(a_t_g_plain, a_t_g, a_t_g1);
        CryptoUtil::intoShares(ah_t_plain, ah_t, ah_t1);
        CryptoUtil::intoShares(z_plain, z, z1);
        CryptoUtil::intoShares(weight_t_plain, weight_t, weight_t1);
        clientTaskComm.sendShareVecVec(a_t_g1, 1 - partyId);
        clientTaskComm.sendShareVecVec(ah_t1, 1 - partyId);
        clientTaskComm.sendShareVecVec(z1, 1 - partyId);
        clientTaskComm.sendShareVecVec(weight_t1, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(a_t_g, 1 - partyId);
        serverTaskComm.recvShareVecVec(ah_t, 1 - partyId);
        serverTaskComm.recvShareVecVec(z, 1 - partyId);
        serverTaskComm.recvShareVecVec(weight_t, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> d;
    std::vector<std::vector<uint64_t>> g;
    sci::twoPartyGCNBackwardNN(a_t_g, ah_t, z, weight_t, normalizer, d, g, false, 1-partyId, role);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> d1;
        std::vector<std::vector<uint64_t>> g1;
        clientTaskComm.recvShareVecVec(d1, 1 - partyId);
        clientTaskComm.recvShareVecVec(g1, 1 - partyId);
        std::vector<std::vector<double>> d_plain;
        std::vector<std::vector<double>> g_plain;
        // sci::print_vector_of_vector(embeddingVecs);
        // sci::print_vector_of_vector(embeddingVecs1);
        CryptoUtil::mergeShareAsDouble(d_plain, d, d1);
        CryptoUtil::mergeShareAsDouble(g_plain, g, g1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected_d);
        sci::print_vector_of_vector(expected_g);
        printf("Actual:\n");
        sci::print_vector_of_vector(d_plain);
        sci::print_vector_of_vector(g_plain);
    } else {
        serverTaskComm.sendShareVecVec(d, 1 - partyId);
        serverTaskComm.sendShareVecVec(g, 1 - partyId);
    }
}

std::vector<std::vector<uint64_t>> generate_random_matrix_uint64(size_t dim0, size_t dim1) {
  srand(42);
  std::vector<std::vector<uint64_t>> result;

  // Loop over the first dimension
  for (size_t i = 0; i < dim0; i++) {
    // Create a std::vector<uint64_t> to store the row
    std::vector<uint64_t> row;
    // Loop over the second dimension
    for (size_t j = 0; j < dim1; j++) {
      // Generate a random element and push it to the row
      uint64_t element = ((((uint64_t)(rand())) << 32) + ((uint64_t)(rand()))) % DataBound;
      row.push_back(element);
    }
    // Push the row to the result
    result.push_back(row);
  }

  // Return the result
  return result;
}

void split_matrix(const std::vector<std::vector<uint64_t>>& matrix, std::vector<std::vector<uint64_t>>& matrix_first, std::vector<std::vector<uint64_t>>& matrix_second) {
  srand(42);

  // Get the dimensions of the matrix
  size_t dim0 = matrix.size();
  size_t dim1 = matrix[0].size();

  // Resize the matrix_first and matrix_second to match the dimensions of the matrix
  matrix_first.resize(dim0);
  matrix_second.resize(dim0);
  for (size_t i = 0; i < dim0; i++) {
    matrix_first[i].resize(dim1);
    matrix_second[i].resize(dim1);
  }

  // Loop over the rows
  for (size_t i = 0; i < dim0; i++) {
    // Loop over the elements
    for (size_t j = 0; j < dim1; j++) {
      // Generate two random numbers that add up to the element mod (1ul<<32)
      uint64_t first = rand() % (Mod);
      uint64_t second = (matrix[i][j] - first) % (Mod);
      // Write the first and second numbers to the matrix_first and matrix_second respectively
      matrix_first[i][j] = first;
      matrix_second[i][j] = second;
    }
  }
}

std::vector<std::vector<uint64_t>> merge_matrix(const std::vector<std::vector<uint64_t>>& matrix_first, const std::vector<std::vector<uint64_t>>& matrix_second) {
  // Get the dimensions of the matrices
  size_t dim0 = matrix_first.size();
  size_t dim1 = matrix_first[0].size();

  // Create a std::vector<std::vector<uint64_t>> to store the result
  std::vector<std::vector<uint64_t>> result;

  // Loop over the rows
  for (size_t i = 0; i < dim0; i++) {
    // Create a std::vector<uint64_t> to store the row
    std::vector<uint64_t> row;
    // Loop over the elements
    for (size_t j = 0; j < dim1; j++) {
      // Compute the sum of the corresponding elements mod (1ul<<32)
      uint64_t element = (matrix_first[i][j] + matrix_second[i][j]) % (Mod);
      // Push the element to the row
      row.push_back(element);
    }
    // Push the row to the result
    result.push_back(row);
  }

  // Return the result
  return result;
}

void test_twoPartyFHEMatMul(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_twoPartyFHEMatMul\n");

    size_t dim0 = 3000;
    size_t dim1 = 1500;
    size_t dim2 = 20;

    // size_t dim0 = 2;
    // size_t dim1 = 3;
    // size_t dim2 = 4;

    // std::vector<std::vector<uint64_t>> A_plain = {
    //     {10, 12, 4},
    //     {1, 2, 3}
    // };
    // std::vector<std::vector<uint64_t>> B_plain = {
    //     {10, 12, 4, 5},
    //     {1, 2, 3, 4},
    //     {4, 9, 1, 1}
    // };

    std::vector<std::vector<uint64_t>> A_plain = generate_random_matrix_uint64(dim0, dim1);
    std::vector<std::vector<uint64_t>> B_plain = generate_random_matrix_uint64(dim1, dim2);
    std::vector<std::vector<uint64_t>> expected_C;
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    troyn::FHEWrapper& local_fhe = cryptoUtil.fhe_crypto;
    expected_C = local_fhe.plainMatMul(A_plain, B_plain);

    std::vector<std::vector<uint64_t>> A;
    std::vector<std::vector<uint64_t>> B;

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> A1;
        std::vector<std::vector<uint64_t>> B1;
        split_matrix(A_plain, A, A1);
        split_matrix(B_plain, B, B1);

        // printf("------> B ALICE Actual\n");
        // sci::print_vector_of_vector(B);
        // printf("<------ B ALICE Actual\n");
        // printf("------> A BOB Actual\n");
        // sci::print_vector_of_vector(A1);
        // printf("<------ A BOB Actual\n");
        // printf("------> cross ALICE Actual\n");
        // sci::print_vector_of_vector(local_fhe.plainMatMul(A1, B));
        // printf("<------ cross ALICE Actual\n");
        // std::vector<std::vector<uint64_t>> C_local_test = local_fhe.plainMatMul(A, B);
        // sci::print_vector_of_vector(C_local_test);
        clientTaskComm.sendShareVecVec(A1, 1 - partyId);
        clientTaskComm.sendShareVecVec(B1, 1 - partyId);
    } else {
        serverTaskComm.recvShareVecVec(A, 1 - partyId);
        serverTaskComm.recvShareVecVec(B, 1 - partyId);
    }

    std::vector<std::vector<uint64_t>> C;
    sci::twoPartyFHEMatMul(A, B, C, 1-partyId, role);

    if (role == sci::ALICE) {
        std::vector<std::vector<uint64_t>> C1;
        clientTaskComm.recvShareVecVec(C1, 1 - partyId);
        auto C_plain = merge_matrix(C, C1);
        printf("Expected:\n");
        sci::print_vector_of_vector(expected_C, 10);
        printf("Actual:\n");
        sci::print_vector_of_vector(C_plain, 10);
    } else {
        serverTaskComm.sendShareVecVec(C, 1 - partyId);
    }
}

void test_FHE() {
    FHEWrapper();

    // size_t dim0 = 200;
    // size_t dim1 = 300;
    // size_t dim2 = 40;

    // // std::vector<std::vector<uint64_t>> A_plain = {
    // //     {10, 12, 4},
    // //     {1, 2, 3}
    // // };
    // // std::vector<std::vector<uint64_t>> B_plain = {
    // //     {10, 12, 4, 5},
    // //     {1, 2, 3, 4},
    // //     {4, 9, 1, 1}
    // // };

    // std::vector<std::vector<uint64_t>> A_plain = generate_random_matrix_uint64(dim0, dim1);
    // std::vector<std::vector<uint64_t>> B_plain = generate_random_matrix_uint64(dim1, dim2);

    // fw.encrypt(B_plain, dim0, dim1, dim2, false);
}

int main(int argc, char* argv[]) {

    int partyNum = 2;
    int partyId = (int)strtol(argv[1], NULL, 10);
    int role;
    if (partyId == 0) role = sci::ALICE;
    else role = sci::BOB;
    int numThreads = 20;

    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    clientTaskComm.tileNumIs(partyNum);
    clientTaskComm.tileIndexIs(partyId);

    printf("HERE1\n");

    TaskComm& serverTaskComm = TaskComm::getServerInstance();
    serverTaskComm.tileNumIs(partyNum);
    serverTaskComm.tileIndexIs(partyId);

    printf("HERE2\n");

    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    cryptoUtil.tileIndexIs(partyId);
    cryptoUtil.setUpPaillierCipher();
    // cryptoUtil.setUpFHECipher();

    printf("HERE3\n");

    std::thread clientSetupThread([&clientTaskComm](){
        clientTaskComm.setUp(true);
    });

    std::thread serverSetupThread([&serverTaskComm](){
        serverTaskComm.setUp(false);
    });

    clientSetupThread.join();
    serverSetupThread.join();

    printf("HERE4\n");

    sci::setUpSCIChannel();
    sci::setUpSCIChannel(role, 1 - partyId);

    test_twoPartyGCNVectorScale(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNSingleVectorScale(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNCondVectorAddition(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNForwardNN(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNForwardNNPrediction(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNBackwardNNInit(clientTaskComm, serverTaskComm, partyId, role);
    test_twoPartyGCNBackwardNN(clientTaskComm, serverTaskComm, partyId, role);

    test_twoPartyFHEMatMul(clientTaskComm, serverTaskComm, partyId, role);

    test_FHE();

    // sci::closeSCIChannel(role, 1 - partyId);

    clientTaskComm.closeChannels();
    serverTaskComm.closeChannels();
    
    return 0;
}