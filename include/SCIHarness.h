#ifndef SCI_HARNESS_H_
#define SCI_HARNESS_H_

#include "task.h"
#include "TaskUtil.h"
#include "TaskqHandler.h"

#include "FloatingPoint/fixed-point.h"
// #include "FloatingPoint/floating-point.h"
// #include "FloatingPoint/fp-math.h"

namespace sci{

uint64_t computeTaskq(vector<Task>& taskq, uint32_t curTid, uint32_t dstTid, uint32_t mpcBasePort, uint32_t tileNum, uint32_t tileIndex, int party, int rotation = 0);

void twoPartyGCNVectorScale(
  const ShareVecVec& _embedding, 
  const std::vector<uint64_t>& _scaler0, 
  const std::vector<uint64_t>& _scaler1, 
  ShareVecVec& output, 
  int dstTid, 
  int party
);

void twoPartyGCNVectorScale(
    const ShareVecVec& _embedding, 
    const std::vector<uint64_t>& _scaler, 
    ShareVecVec& output, 
    bool _isSigned,
    int dstTid, 
    int party
);

void twoPartyGCNMatrixScale(
  const ShareVecVec& _embedding, 
  uint64_t _scaler, 
  ShareVecVec& output, 
  int dstTid, 
  int party
);

void twoPartyGCNApplyGradient(
  const ShareVecVec& _weight,
  const ShareVecVec& _gradient, 
  uint64_t _lr, 
  ShareVecVec& output, 
  int dstTid, 
  int party
);

void twoPartyGCNCondVectorAddition(
  const ShareVecVec& _vec0, 
  const ShareVecVec& _vec1, 
  const std::vector<bool>& _cond, 
  ShareVecVec& output, 
  int dstTid, 
  int party
);

void twoPartyMux2(
  const std::vector<uint64_t>& input1,
  const std::vector<uint64_t>& input2,
  const std::vector<uint8_t>& selector,
  std::vector<uint64_t>& result,
  int dstTid,
  int party,
  bool is_one_side = false
);

void twoPartyMux2(
  const std::vector<std::vector<uint64_t>>& input1,
  const std::vector<std::vector<uint64_t>>& input2,
  const std::vector<uint8_t>& selector,
  std::vector<std::vector<uint64_t>>& result,
  int dstTid,
  int party,
  bool is_one_side = false
);

void twoPartyEq(
    const std::vector<uint64_t>& lhs,
    const std::vector<uint64_t>& rhs,
    std::vector<uint8_t>& result,
    int dstTid, 
    int party
);

void twoPartyFHEMatMul(
    const ShareVecVec& A,
    const ShareVecVec& B,
    ShareVecVec& C,
    int dstTid,
    int party
);

void twoPartyGCNForwardNN(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _weight, 
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _z, 
    ShareVecVec& _new_h, 
    int dstTid, 
    int party
);

void twoPartyGCNForwardNNPrediction(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _weight, 
    const ShareVecVec& _y,
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _z, 
    ShareVecVec& _p, 
    ShareVecVec& _p_minus_y, 
    int dstTid, 
    int party
);

void twoPartyGCNBackwardNNInit(
    const ShareVecVec& _p_minus_y,
    const ShareVecVec& _ah_t,
    const ShareVecVec& _weight_t,
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _d,
    ShareVecVec& _g,
    int dstTid,
    int party
);

void twoPartyGCNBackwardNN(
    const ShareVecVec& _a_t_g,
    const ShareVecVec& _ah_t,
    const ShareVecVec& _z,
    const ShareVecVec& _weight_t,
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _d,
    ShareVecVec& _g,
    bool isFirstLayer,
    int dstTid,
    int party
);

void twoPartyGCNMatMul(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _weight, 
    ShareVecVec& _z, 
    int dstTid, 
    int party
);

void twoPartyGCNRelu(
    const ShareVecVec& _embedding, 
    ShareVecVec& _new_h, 
    int dstTid, 
    int party
);

void twoPartyGCNForwardNNPredictionWithoutWeight(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _y,
    ShareVecVec& _p, 
    ShareVecVec& _p_minus_y, 
    int dstTid, 
    int party
);

void twoPartyGCNBackwardNNWithoutAH(
    const ShareVecVec& _a_t_g,
    const ShareVecVec& _z,
    const ShareVecVec& _weight_t,
    ShareVecVec& _dz_a_t_g,
    ShareVecVec& _g,
    bool isFirstLayer,
    int dstTid,
    int party
);

void twoPartyCmpSwap(std::vector<uint64_t>& lhs, std::vector<uint64_t>& rhs, int partyId, int party, int threadId = 0);
void twoPartyCmpSwap(std::vector<std::vector<uint64_t>>& lhs, std::vector<std::vector<uint64_t>>& rhs, int partyId, int party, int threadId = 0);
void twoPartyCmpOpen(std::vector<uint64_t>& lhs, std::vector<uint64_t>& rhs, std::vector<bool>& result, int partyId, int party, int threadId = 0);
void twoPartySelectedAssign(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, const std::vector<uint8_t>& selector, int partyId, int party, bool is_one_side = false, int threadId = 0);
void twoPartyAdd(const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs, std::vector<uint64_t>& result, int partyId, int party, int threadId = 0);

void printShareVecVec(const ShareVecVec& svv, int dstTid, int party);
void getPlainShareVecVec(const ShareVecVec& svv, DoubleTensor& result, int dstTid, int party);
double cross_entropy_loss(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred);
double accuracy(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred);
double accuracy(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred, const std::vector<bool>& is_border);

void setUpSCIChannel();
void closeSCIChannel();

void setUpSCIChannel(int party, uint32_t dstTid);
void closeSCIChannel(int party, uint32_t dstTid);

template <typename T>
void print_vector(const std::vector<T>& v, size_t num = 0) {
  size_t cnt = 0;
  for (auto elem : v) {
    if (num != 0 && cnt >= num) break;
    std::cout << elem << " ";
    cnt++;
  }
  std::cout << "\n";
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

template <typename T>
void plaintext_add_matrix_in_place(std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
  size_t dim0 = a.size();
  size_t dim1 = a[0].size();
  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      a[i][j] += b[i][j];
    }
  }
}

template <typename T>
std::vector<std::vector<T>> plaintext_add_matrix(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
  size_t dim0 = a.size();
  size_t dim1 = a[0].size();
  std::vector<std::vector<T>> ret = a;
  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      ret[i][j] += b[i][j];
    }
  }
  return ret;
}

template <typename T>
std::vector<std::vector<T>> plaintext_sub_matrix(const std::vector<std::vector<T>>& a, const std::vector<std::vector<T>>& b) {
  size_t dim0 = a.size();
  size_t dim1 = a[0].size();
  std::vector<std::vector<T>> ret = a;
  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      ret[i][j] -= b[i][j];
    }
  }
  return ret;
}

std::string pack_string_vec(const std::vector<std::string>& sv, std::vector<size_t>& sizev);
std::vector<std::string> unpack_string_vec(const std::string& packed, const std::vector<size_t>& sizev);

uint64_t count_true(const std::vector<bool>& vec);

} // namespace sci


#endif