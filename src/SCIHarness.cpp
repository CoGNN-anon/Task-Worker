#include "SCIHarness.h"
#include <iostream>
#include <string>
#include <mutex>
#include <cassert>

namespace sci{

const uint64_t MultDataBound = (1ul << 44);
const uint64_t MultMod = (1ul << 44);
const uint64_t MultModBitLength = 44;
const uint64_t MultScaler = (1ul << SCALER_BIT_LENGTH);

std::vector<NetIO*> ALICE_iopacks;
std::vector<NetIO*> BOB_iopacks;
std::vector<OTPack<NetIO>*> ALICE_otpacks;
std::vector<OTPack<NetIO>*> BOB_otpacks;
std::vector<FixOp*> ALICE_fix_ops;
std::vector<FixOp*> BOB_fix_ops;

std::mutex print_comm_mutex;

uint64_t get_comm(uint32_t dstTid, int party) {
    uint64_t c_byte = 0;
    print_comm_mutex.lock();

    print_comm_mutex.unlock();
    return c_byte;
}

void print_comm(uint64_t c_byte0, uint32_t dstTid, int party, std::string tag) {
    auto c_mbyte = ((double)get_comm(dstTid, party) - c_byte0) / (1024 * 1024);
    std::cout << ">>" << tag << " comm " << c_mbyte << " MB\n";
}

#define print_fixed(vec)                                                         \
  {                                                                            \
    auto tmp_pub = fix_op->output(PUBLIC, vec).subset(I, I + J);                  \
    std::cout << #vec << "_pub: " << tmp_pub << std::endl;                               \
  }

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n) {
   auto first = v.begin() + m;
   auto last = v.begin() + n;
   std::vector<T> vector(first, last);
   return vector;
}

uint64_t computeTaskq(std::vector<Task>& taskq, uint32_t curTid, uint32_t dstTid, uint32_t mpcBasePort, uint32_t tileNum, uint32_t tileIndex, int party, int rotation) {
    return 0;
}

void twoPartyGCNVectorScale(
    const ShareVecVec& _embedding, 
    const std::vector<uint64_t>& _scaler0, 
    const std::vector<uint64_t>& _scaler1, 
    ShareVecVec& output, 
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t taskNum = _embedding.size();
    size_t embedSize = _embedding[0].size();

    std::vector<FixArray> embedding(taskNum);
    std::vector<FixArray> scaler0(taskNum);
    std::vector<FixArray> scaler1(taskNum);
    for (int i=0; i<taskNum; ++i) {
        embedding[i] = fix_op->input(party, embedSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
        embedding[i] = fix_op->reduce(embedding[i], 32);
        scaler0[i] = fix_op->input(party, embedSize, _scaler0[i], true, 64, SCALER_BIT_LENGTH);
        scaler0[i] = fix_op->reduce(scaler0[i], 32);
        scaler1[i] = fix_op->input(party, embedSize, _scaler1[i], true, 64, SCALER_BIT_LENGTH);
        scaler1[i] = fix_op->reduce(scaler1[i], 32);
    }

    std::vector<FixArray> result_embedding;
    const size_t max_step = 1000;
    for (int lb = 0; lb < taskNum; lb += max_step) {
        int rb = lb + max_step < taskNum ? lb + max_step : taskNum;
        std::vector<FixArray> cur_embedding(embedding.begin() + lb, embedding.begin() + rb);
        std::vector<FixArray> cur_scaler0(scaler0.begin() + lb, scaler0.begin() + rb);
        std::vector<FixArray> cur_scaler1(scaler1.begin() + lb, scaler1.begin() + rb);
        FixArray cur_embedding_flat = concat(cur_embedding);
        FixArray cur_scaler0_flat = concat(cur_scaler0);
        FixArray cur_scaler1_flat = concat(cur_scaler1);
        cur_embedding_flat = fix_op->mul(cur_embedding_flat, cur_scaler0_flat, 64);
        cur_embedding_flat = fix_op->truncate_reduce(cur_embedding_flat, SCALER_BIT_LENGTH);
        cur_embedding_flat = fix_op->reduce(cur_embedding_flat, 32);
        cur_embedding_flat = fix_op->mul(cur_embedding_flat, cur_scaler1_flat, 64);
        cur_embedding_flat = fix_op->truncate_reduce(cur_embedding_flat, SCALER_BIT_LENGTH);
        cur_embedding_flat = fix_op->extend(cur_embedding_flat, 64);
        cur_embedding = deConcat(cur_embedding_flat, rb - lb, embedSize);
        result_embedding.insert(result_embedding.end(), cur_embedding.begin(), cur_embedding.end());
    }

    assert(result_embedding.size() == taskNum);

    // FixArray embedding_flat = concat(embedding);
    // FixArray scaler0_flat = concat(scaler0);
    // FixArray scaler1_flat = concat(scaler1);
    
// #ifndef NDEBUG
//     {
//         std::cout << osuCrypto::IoStream::lock;
//         printf("Vector Scale Input\n");
//         printf("----------->\n");
//         for (int i = 0; i < taskNum; ++i) fix_op->print(embedding[i]);
//         printf("-----------\n");
//         for (int i = 0; i < taskNum; ++i) fix_op->print(scaler0[i]);
//         printf("-----------\n");
//         for (int i = 0; i < taskNum; ++i) fix_op->print(scaler1[i]);
//         printf("<-----------\n");
//         std::cout << osuCrypto::IoStream::unlock;
//     }
// #endif

    // embedding_flat = fix_op->mul(embedding_flat, scaler0_flat, 64);
    // embedding_flat = fix_op->truncate_reduce(embedding_flat, SCALER_BIT_LENGTH);
    // embedding_flat = fix_op->reduce(embedding_flat, 32);
    // embedding_flat = fix_op->mul(embedding_flat, scaler1_flat, 64);
    // embedding_flat = fix_op->truncate_reduce(embedding_flat, SCALER_BIT_LENGTH);
    // embedding_flat = fix_op->extend(embedding_flat, 64);

    // embedding = deConcat(embedding_flat, taskNum, embedSize);    

// #ifndef NDEBUG
//     {
//         std::cout << osuCrypto::IoStream::lock;
//         printf("Vector Scale output\n");
//         printf("----------->\n");
//         for (int i = 0; i < taskNum; ++i) fix_op->print(embedding[i]);
//         printf("<-----------\n");
//         std::cout << osuCrypto::IoStream::unlock;
//     }
// #endif

    output.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        output[i].resize(embedSize);
        output[i].assign((uint64_t*)result_embedding[i].data, (uint64_t*)result_embedding[i].data+embedSize);
    }
}

void twoPartyGCNMatrixScale(
  const ShareVecVec& _embedding, 
  uint64_t _scaler, 
  ShareVecVec& output, 
  int dstTid, 
  int party
) {
    auto t_tmp = std::chrono::high_resolution_clock::now();

    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t taskNum = _embedding.size();
    size_t embedSize = _embedding[0].size();

    std::vector<FixArray> embedding(taskNum);
    for (int i=0; i<taskNum; ++i) {
        embedding[i] = fix_op->input(party, embedSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
        embedding[i] = fix_op->reduce(embedding[i], 32);
    }

    FixArray embedding_flat = concat(embedding);

    embedding_flat = fix_op->mul(embedding_flat, _scaler, 64);
    embedding_flat.s = 2 * SCALER_BIT_LENGTH;
    embedding_flat = fix_op->truncate_reduce(embedding_flat, SCALER_BIT_LENGTH);
    embedding_flat = fix_op->extend(embedding_flat, 64);

    embedding = deConcat(embedding_flat, taskNum, embedSize);

    output.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        output[i].resize(embedSize);
        output[i].assign((uint64_t*)embedding[i].data, (uint64_t*)embedding[i].data+embedSize);
    }    

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNMatrixScale");
}

void twoPartyFHEVectorScale(
    const ShareVecVec& A,
    const std::vector<uint64_t>& S,
    ShareVecVec& C,
    int dstTid,
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    troyn::FHEWrapper& local_fhe = cryptoUtil.fhe_crypto;
    troyn::FHEWrapper& remote_fhe = *(cryptoUtil.remote_fhe_crypto.find(dstTid)->second);

    size_t dim0 = A.size();
    size_t dim1 = A[0].size();
    assert(S.size() == dim0);

    ShareVecVec A_reduce = A;
    std::vector<uint64_t> S_reduce = S;

    for (int i = 0; i < dim0; ++i) {
        S_reduce[i] %= MultMod;
        for (int j = 0; j < dim1; ++j) A_reduce[i][j] %= MultMod;
    }

    if (party == sci::ALICE) {
        TaskComm& taskComm = TaskComm::getClientInstance();
        osuCrypto::Channel& chl = *taskComm.getChannel(dstTid);

        std::vector<std::string> remote_enc_A;
        std::string packed_remote_enc_A;
        std::vector<size_t> packed_remote_size_A;
        chl.recv(packed_remote_enc_A);
        chl.recv(packed_remote_size_A);
        remote_enc_A = unpack_string_vec(packed_remote_enc_A, packed_remote_size_A);
        C.resize(0);
        cryptoUtil.fhe_mut.lock();
        for (int i = 0; i < dim0; ++i) C.push_back(remote_fhe.add_and_mul_scaler_and_subtract_random(remote_enc_A[i], A_reduce[i], S_reduce[i], dim1));
        cryptoUtil.fhe_mut.unlock();
        packed_remote_enc_A = pack_string_vec(remote_enc_A, packed_remote_size_A);
        chl.asyncSendCopy(packed_remote_enc_A);
        chl.asyncSendCopy(packed_remote_size_A);
    } else {
        TaskComm& taskComm = TaskComm::getServerInstance();
        osuCrypto::Channel& chl = *taskComm.getChannel(dstTid);

        std::vector<std::string> enc_A;
        cryptoUtil.fhe_mut.lock();
        for (int i = 0; i < dim0; ++i) enc_A.push_back(local_fhe.encrypt(A_reduce[i]));
        cryptoUtil.fhe_mut.unlock();

        std::string packed_enc_A;
        std::vector<size_t> packed_size_A;
        packed_enc_A = pack_string_vec(enc_A, packed_size_A);
        chl.asyncSendCopy(packed_enc_A);
        chl.asyncSendCopy(packed_size_A);

        std::vector<std::string> enc_C;
        std::string packed_enc_C;
        std::vector<size_t> packed_size_C;
        chl.recv(packed_enc_C);
        chl.recv(packed_size_C);
        enc_C = unpack_string_vec(packed_enc_C , packed_size_C);
        C.resize(0);
        cryptoUtil.fhe_mut.lock();
        for (int i = 0; i < dim0; ++i) C.push_back(local_fhe.decrypt(enc_C[i], dim1));
        cryptoUtil.fhe_mut.unlock();
    }
}

void twoPartyGCNVectorScale(
    const ShareVecVec& _embedding, 
    const std::vector<uint64_t>& _scaler, 
    ShareVecVec& output, 
    bool _isSigned,
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t taskNum = _embedding.size();
    size_t embedSize = _embedding[0].size();

    if (embedSize >= 500) {
        auto t_tmp = std::chrono::high_resolution_clock::now();  
        ShareVecVec _scaled;
        twoPartyFHEVectorScale(
            _embedding,
            _scaler,
            _scaled,
            dstTid,
            party
        );    
        if (party == sci::ALICE) print_duration(t_tmp, "fhe-svv-scale");
        t_tmp = std::chrono::high_resolution_clock::now();  

        std::vector<FixArray> scaled(taskNum);
        for (int i=0; i<taskNum; ++i) {
            scaled[i] = fix_op->input(party, embedSize, (uint64_t*)&(_scaled[i][0]), _isSigned, MultModBitLength, 2*SCALER_BIT_LENGTH);
        }
        FixArray scaled_flat = concat(scaled);
        scaled_flat = fix_op->truncate_reduce(scaled_flat, SCALER_BIT_LENGTH);
        if (_isSigned) scaled_flat = fix_op->extend(scaled_flat, 64);
        else {
            uint8_t* zero_msb = new uint8_t[taskNum * embedSize]{0};
            scaled_flat = fix_op->extend(scaled_flat, 64, zero_msb);
            delete zero_msb;
        }
        scaled = deConcat(scaled_flat, taskNum, embedSize);
        if (party == sci::ALICE) print_duration(t_tmp, "extend-svv-scale");   

        output.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            output[i].resize(embedSize);
            output[i].assign((uint64_t*)scaled[i].data, (uint64_t*)scaled[i].data+embedSize);
        }
    } else {
        auto t_tmp = std::chrono::high_resolution_clock::now();
        std::vector<FixArray> embedding(taskNum);
        std::vector<FixArray> scaler(taskNum);
        for (int i=0; i<taskNum; ++i) {
            embedding[i] = fix_op->input(party, embedSize, (uint64_t*)&(_embedding[i][0]), _isSigned, 64, SCALER_BIT_LENGTH);
            embedding[i] = fix_op->reduce(embedding[i], 32);
            scaler[i] = fix_op->input(party, embedSize, _scaler[i], _isSigned, 64, SCALER_BIT_LENGTH);
            scaler[i] = fix_op->reduce(scaler[i], 32);
        }

        FixArray embedding_flat = concat(embedding);
        FixArray scaler_flat = concat(scaler);

        // embedding_flat = fix_op->mul(embedding_flat, scaler_flat, 64);
        embedding_flat = fix_op->one_side_mul(embedding_flat, scaler_flat, 64, false, true);
        embedding_flat = fix_op->truncate_reduce(embedding_flat, SCALER_BIT_LENGTH);
        if (_isSigned) embedding_flat = fix_op->extend(embedding_flat, 64);
        else {
            uint8_t* zero_msb = new uint8_t[taskNum * embedSize]{0};
            embedding_flat = fix_op->extend(embedding_flat, 64, zero_msb);
            delete zero_msb;            
        }
        embedding = deConcat(embedding_flat, taskNum, embedSize);   

        output.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            output[i].resize(embedSize);
            output[i].assign((uint64_t*)embedding[i].data, (uint64_t*)embedding[i].data+embedSize);
        }
        if (party == sci::ALICE) print_duration(t_tmp, "OT-svv-scale");
    }
}

void twoPartyGCNApplyGradient(
  const ShareVecVec& _weight,
  const ShareVecVec& _gradient, 
  uint64_t _lr, 
  ShareVecVec& output, 
  int dstTid, 
  int party
) {
    auto t_tmp = std::chrono::high_resolution_clock::now();

    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t dim0 = _weight.size();
    size_t dim1 = _weight[0].size();

    std::vector<FixArray> weight(dim0);
    std::vector<FixArray> gradient(dim0);
    for (int i=0; i<dim0; ++i) {
        weight[i] = fix_op->input(party, dim1, (uint64_t*)&(_weight[i][0]), true, 64, SCALER_BIT_LENGTH);
        gradient[i] = fix_op->input(party, dim1, (uint64_t*)&(_gradient[i][0]), true, 64, SCALER_BIT_LENGTH);
        // gradient[i] = fix_op->reduce(gradient[i], 32);
    }

    FixArray weight_flat = concat(weight);
    FixArray gradient_flat = concat(gradient);

    gradient_flat = fix_op->mul(gradient_flat, _lr, 64);
    gradient_flat.s = 2 * SCALER_BIT_LENGTH;
    gradient_flat = fix_op->truncate_reduce(gradient_flat, SCALER_BIT_LENGTH);
    gradient_flat = fix_op->extend(gradient_flat, 64);
    weight_flat = fix_op->sub(weight_flat, gradient_flat);

    weight = deConcat(weight_flat, dim0, dim1);

    output.resize(dim0);
    for (int i=0; i<dim0; ++i) {
        output[i].resize(dim1);
        output[i].assign((uint64_t*)weight[i].data, (uint64_t*)weight[i].data+dim1);
    }    

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNApplyGradient");
}

void twoPartyMux2(
    const std::vector<uint64_t>& input1, 
    const std::vector<uint64_t>& input2, 
    const std::vector<uint8_t>& selector, 
    std::vector<uint64_t>& result, 
    int dstTid, 
    int party, 
    bool is_one_side
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t length = input1.size();
    assert(input2.size() == length);
    assert(selector.size() == length);
    FixArray sci_input1 = fix_op->input(party, length, (uint64_t*)input1.data(), true, 64, SCALER_BIT_LENGTH);
    FixArray sci_input2 = fix_op->input(party, length, (uint64_t*)input2.data(), true, 64, SCALER_BIT_LENGTH);
    BoolArray sci_selector = fix_op->bool_op->input(party, length, (uint8_t*)selector.data());

    FixArray sci_result;
    if (is_one_side) {
        sci_result = fix_op->one_side_if_else(sci_selector, sci_input2, sci_input1);
        // sci_result = fix_op->if_else(sci_selector, sci_input2, sci_input1);
    } else {
        sci_result = fix_op->if_else(sci_selector, sci_input2, sci_input1);
    }

    // sci_result = fix_op->output(emp::PUBLIC, sci_result);
    
    result.resize(length);
    for (int i = 0; i < length; ++i) {
        result[i] = sci_result.data[i];
    }    
}

void twoPartyMux2(
    const std::vector<std::vector<uint64_t>>& input1, 
    const std::vector<std::vector<uint64_t>>& input2, 
    const std::vector<uint8_t>& selector, 
    std::vector<std::vector<uint64_t>>& result, 
    int dstTid, 
    int party, 
    bool is_one_side
) {
    size_t dim0 = input1.size();
    size_t dim1 = input1[0].size();
    size_t flat_dim = dim0 * dim1;
    std::vector<uint64_t> flat_input1(flat_dim);
    std::vector<uint64_t> flat_input2(flat_dim);
    std::vector<uint64_t> flat_result(flat_dim);
    std::vector<uint8_t> flat_selector(flat_dim);
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            flat_input1[i * dim1 + j] = input1[i][j];
            flat_input2[i * dim1 + j] = input2[i][j];
            flat_selector[i * dim1 + j] = selector[i];
        }
    }

    twoPartyMux2(flat_input1, flat_input2, flat_selector, flat_result, dstTid, party, is_one_side);
    
    result.resize(dim0, std::vector<uint64_t>(dim1, 0));
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            result[i][j] = flat_result[i * dim1 + j];
        }
    }    
}

void twoPartyEq(
    const std::vector<uint64_t>& lhs,
    const std::vector<uint64_t>& rhs,
    std::vector<uint8_t>& result,
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    size_t length = lhs.size();
    FixArray sci_lhs = fix_op->input(party, length, (uint64_t*)lhs.data(), true, 64, SCALER_BIT_LENGTH);
    FixArray sci_rhs = fix_op->input(party, length, (uint64_t*)rhs.data(), true, 64, SCALER_BIT_LENGTH);

    // Compute
    BoolArray cmp_result = fix_op->EQ(sci_lhs, sci_rhs);

    // printf("h1\n");
    // cmp_result = fix_op->bool_op->output(emp::PUBLIC, cmp_result);
    // printf("h2\n");
    
    result.resize(length);
    for (int i = 0; i < length; ++i) {
        result[i] = cmp_result.data[i];
    }    
}

void twoPartyGCNCondVectorAddition(
  const ShareVecVec& _vec0, 
  const ShareVecVec& _vec1, 
  const std::vector<bool>& _cond, 
  ShareVecVec& output, 
  int dstTid, 
  int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    
    uint64_t taskNum = _vec0.size();
    size_t vecSize = _vec0[0].size();

    std::vector<FixArray> sci_operands[3];
    FixArray sci_operands_flat[3];
    for (int i=0; i<3; ++i) sci_operands[i].resize(taskNum);
    std::vector<uint8_t> extend_operandMask(taskNum*vecSize, 0);
    BoolArray sci_operandMask_flat;

    for (int i=0; i<taskNum; ++i) {
        sci_operands[0][i] = fix_op->input(party, vecSize, (uint64_t*)&(_vec0[i][0]), true, 64, SCALER_BIT_LENGTH);
        // sci_operands[0][i] = fix_op->reduce(sci_operands[0][i], 32);
        sci_operands[1][i] = fix_op->input(party, vecSize, (uint64_t*)&(_vec1[i][0]), true, 64, SCALER_BIT_LENGTH);
        // sci_operands[1][i] = fix_op->reduce(sci_operands[1][i], 32);

        uint8_t operandMask;
        if (_cond[i]) operandMask = 1;
        else operandMask = 0;
        int32_t offset = i*vecSize;
        for (int j=0; j<vecSize; ++j) extend_operandMask[offset+j] = operandMask;
    }

    sci_operands_flat[0] = concat(sci_operands[0]);
    sci_operands_flat[1] = concat(sci_operands[1]);
    sci_operands_flat[2] = fix_op->add(sci_operands_flat[0], sci_operands_flat[1]);
    
    // sci_operands_flat[0] = fix_op->extend(sci_operands_flat[0], 64);
    // sci_operands_flat[2] = fix_op->extend(sci_operands_flat[2], 64);

    sci_operandMask_flat = fix_op->bool_op->input(ALICE, taskNum*vecSize, (uint8_t*)&extend_operandMask[0]);
    sci_operands_flat[0] = fix_op->one_side_if_else(sci_operandMask_flat, sci_operands_flat[2], sci_operands_flat[0]);

    sci_operands[0] = deConcat(sci_operands_flat[0], taskNum, vecSize);

    output = _vec0;
    for (int i=0; i<taskNum; ++i) {
        output[i].assign((uint64_t*)sci_operands[0][i].data, (uint64_t*)sci_operands[0][i].data+vecSize);
    }    
}

void twoPartyFHEAtomicMatMul(
    const ShareVecVec& A,
    const ShareVecVec& B,
    ShareVecVec& C,
    int dstTid,
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    troyn::FHEWrapper& local_fhe = cryptoUtil.fhe_crypto;
    troyn::FHEWrapper& remote_fhe = *(cryptoUtil.remote_fhe_crypto.find(dstTid)->second);

    size_t dim0 = A.size();
    size_t dim1 = A[0].size();
    size_t dim2 = B[0].size();

    ShareVecVec A_reduce = A;
    ShareVecVec B_reduce = B;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) A_reduce[i][j] %= MultMod;
    }
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) B_reduce[i][j] %= MultMod;
    }

    ShareVecVec local_C = local_fhe.plainMatMul(A, B);
    // print_vector_of_vector(local_C);
    ShareVecVec local_cross0;
    ShareVecVec local_cross1;

    if (party == sci::ALICE) {
        TaskComm& taskComm = TaskComm::getClientInstance();
        osuCrypto::Channel& chl = *taskComm.getChannel(dstTid);

        cryptoUtil.fhe_mut.lock();
        std::string enc_B = local_fhe.encrypt(B_reduce, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
        // printf("remote_enc_B_size 1 %lu \n", enc_B.size());
        chl.asyncSendCopy(enc_B);
        // size_t enc_B_size = enc_B.size();
        // fix_op->iopack->io->send_data(&enc_B_size, sizeof(size_t));
        // fix_op->iopack->io->send_data(enc_B.c_str(), enc_B_size);
        std::string remote_enc_B;
        // size_t remote_enc_B_size = 0;
        // fix_op->iopack->io->recv_data(&remote_enc_B_size, sizeof(size_t));
        // char *_remote_enc_B = new char[remote_enc_B_size + 1];
        // fix_op->iopack->io->recv_data(_remote_enc_B, remote_enc_B_size);
        // remote_enc_B = std::string(_remote_enc_B, remote_enc_B_size);
        // printf("remote_enc_B size 2 %lu\n", remote_enc_B.size());
        // delete _remote_enc_B;
        chl.recv(remote_enc_B);
        // std::cout<<"Second: "<<std::endl;
        // for (int i = 0; i < 10; i++) {
        //     // Get the ASCII value of the character at position i
        //     int ascii = (int)remote_enc_B[i];

        //     // Print the hexadecimal representation of the ASCII value
        //     printf("%x ", ascii);
        // }
        // printf("------> B ALICE\n");
        // print_vector_of_vector(B);
        // printf("<------ B ALICE\n");
        cryptoUtil.fhe_mut.lock();
        local_cross0 = remote_fhe.mat_mul_and_subtract_random(remote_enc_B, A_reduce, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
        chl.asyncSendCopy(remote_enc_B);

        // remote_enc_B_size = remote_enc_B.size();
        // fix_op->iopack->io->send_data(&remote_enc_B_size, sizeof(size_t));
        // fix_op->iopack->io->send_data(remote_enc_B.c_str(), remote_enc_B_size);

        std::string remote_enc_C;
        chl.recv(remote_enc_C);
        // size_t remote_enc_C_size = 0;
        // fix_op->iopack->io->recv_data(&remote_enc_C_size, sizeof(size_t));
        // char *_remote_enc_C = new char[remote_enc_C_size + 1];
        // fix_op->iopack->io->recv_data(_remote_enc_C, remote_enc_C_size);
        // remote_enc_C = std::string(_remote_enc_C, remote_enc_C_size);
        // delete _remote_enc_C;
        cryptoUtil.fhe_mut.lock();
        local_cross1 = local_fhe.decrypt(remote_enc_C, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
        // printf("------> cross ALICE\n");
        // print_vector_of_vector(local_cross1);
        // printf("<------ cross ALICE\n");
    } else {
        TaskComm& taskComm = TaskComm::getServerInstance();
        osuCrypto::Channel& chl = *taskComm.getChannel(dstTid);

        cryptoUtil.fhe_mut.lock();
        std::string enc_B = local_fhe.encrypt(B_reduce, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
        chl.asyncSendCopy(enc_B);
        // size_t enc_B_size = enc_B.size();
        // fix_op->iopack->io->send_data(&enc_B_size, sizeof(size_t));
        // fix_op->iopack->io->send_data(enc_B.c_str(), enc_B_size);

        std::string remote_enc_B;
        // size_t remote_enc_B_size = 0;
        // fix_op->iopack->io->recv_data(&remote_enc_B_size, sizeof(size_t));
        // char *_remote_enc_B = new char[remote_enc_B_size + 1];
        // fix_op->iopack->io->recv_data(_remote_enc_B, remote_enc_B_size);
        // remote_enc_B = std::string(_remote_enc_B, remote_enc_B_size);
        // delete _remote_enc_B;
        chl.recv(remote_enc_B);
        // printf("------> A BOB\n");
        // print_vector_of_vector(A);
        // printf("<------ A BOB\n");
        cryptoUtil.fhe_mut.lock();
        local_cross0 = remote_fhe.mat_mul_and_subtract_random(remote_enc_B, A_reduce, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
        chl.asyncSendCopy(remote_enc_B);
        // remote_enc_B_size = remote_enc_B.size();
        // fix_op->iopack->io->send_data(&remote_enc_B_size, sizeof(size_t));
        // fix_op->iopack->io->send_data(remote_enc_B.c_str(), remote_enc_B_size);

        std::string remote_enc_C;
        chl.recv(remote_enc_C);
        // size_t remote_enc_C_size = 0;
        // fix_op->iopack->io->recv_data(&remote_enc_C_size, sizeof(size_t));
        // char *_remote_enc_C = new char[remote_enc_C_size + 1];
        // fix_op->iopack->io->recv_data(_remote_enc_C, remote_enc_C_size);
        // remote_enc_C = std::string(_remote_enc_C, remote_enc_C_size);
        // delete _remote_enc_C;
        cryptoUtil.fhe_mut.lock();
        local_cross1 = local_fhe.decrypt(remote_enc_C, dim0, dim1, dim2, false);
        cryptoUtil.fhe_mut.unlock();
    }

    C.resize(dim0, std::vector<uint64_t>(dim2, 0));
    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            C[i][j] = (local_C[i][j] + local_cross0[i][j] + local_cross1[i][j]) % MultMod;
        }
    }
}

void twoPartyFHEMatMul(
    const ShareVecVec& A,
    const ShareVecVec& B,
    ShareVecVec& C,
    int dstTid,
    int party
) {
    size_t A_length = A.size();
    const size_t max_step = 3000 * 1000 / A[0].size();
    
    ShareVecVec result_C;
    for (int lb = 0; lb < A_length; lb += max_step) {
        int rb = lb + max_step < A_length ? lb + max_step : A_length;
        ShareVecVec cur_A(A.begin() + lb, A.begin() + rb);
        ShareVecVec cur_C;
        twoPartyFHEAtomicMatMul(
            cur_A,
            B,
            cur_C,
            dstTid,
            party
        );
        result_C.insert(result_C.end(), cur_C.begin(), cur_C.end());
    }
    assert(result_C.size() == A_length);
    C.swap(result_C);
    return;
}

void twoPartyGCNForwardNN(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _weight, 
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _z, 
    ShareVecVec& _new_h, 
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    uint64_t taskNum = _embedding.size();
    size_t hSize = _embedding[0].size();
    if (_weight.size() != hSize) {
        printf("Unmatched weight matrix and vector dimensions during ssGCNForwardNN, _weight.size() = %lu, hSize = %lu\n", _weight.size(), hSize);
        exit(-1);
    }
    size_t new_hSize = _weight[0].size();

    // std::vector<FixArray> z = fix_op->mul(embedding, weight, 32);

    // std::vector<FixArray> weight;
    // weight.resize(hSize);
    // for (int i=0; i<hSize; ++i) {
    //     weight[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_weight[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     weight[i] = fix_op->reduce(weight[i], 32);
    // }

    // std::vector<FixArray> embedding(taskNum);
    // for (int i=0; i<taskNum; ++i) {
    //     embedding[i] = fix_op->input(party, hSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     embedding[i] = fix_op->reduce(embedding[i], 32);
    // }

    auto t_tmp = std::chrono::high_resolution_clock::now(); 

    ShareVecVec _z_reduce;
    twoPartyFHEMatMul(_embedding, _weight, _z_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-mat-mul");
    t_tmp = std::chrono::high_resolution_clock::now(); 

    std::vector<FixArray> z;
    z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        z[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_z_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray z_flat = concat(z);
    z_flat = fix_op->truncate_reduce(z_flat, SCALER_BIT_LENGTH);
    z_flat = fix_op->extend(z_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-truncate-and-extend");
    t_tmp = std::chrono::high_resolution_clock::now();

    // Scale
    if (_normalizer.size() != 0) {
        _z.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
            _z[i].assign((uint64_t*)z_offset, (uint64_t*)z_offset+new_hSize);
        }
        twoPartyGCNVectorScale(
            _z, 
            _normalizer, 
            _z, 
            true,
            dstTid, 
            party
        );
        for (int i=0; i<taskNum; ++i) {
            uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
            for (int j=0; j<new_hSize; ++j) *(z_offset+j) =  _z[i][j];
        }
    }    

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-vector-scale");
    t_tmp = std::chrono::high_resolution_clock::now();

    FixArray new_h_flat = fix_op->relu(z_flat);
    new_h_flat = fix_op->extend(new_h_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-relu");

    _z.resize(taskNum);
    _new_h.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
        uint64_t* new_h_offset = (uint64_t*)new_h_flat.data + i*new_hSize;
        _z[i].assign((uint64_t*)z_offset, (uint64_t*)z_offset+new_hSize);
        _new_h[i].assign((uint64_t*)new_h_offset, (uint64_t*)new_h_offset+new_hSize);
    } 
}

void twoPartyGCNMatMul(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _weight, 
    ShareVecVec& _z, 
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    uint64_t taskNum = _embedding.size();
    size_t hSize = _embedding[0].size();
    if (_weight.size() != hSize) {
        printf("Unmatched weight matrix and vector dimensions during gcn mat mul, _weight.size() = %lu, hSize = %lu\n", _weight.size(), hSize);
        exit(-1);
    }
    size_t new_hSize = _weight[0].size();

    auto t_tmp = std::chrono::high_resolution_clock::now(); 

    ShareVecVec _z_reduce;
    twoPartyFHEMatMul(_embedding, _weight, _z_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-mat-mul");
    t_tmp = std::chrono::high_resolution_clock::now(); 

    std::vector<FixArray> z;
    z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        z[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_z_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray z_flat = concat(z);
    z_flat = fix_op->truncate_reduce(z_flat, SCALER_BIT_LENGTH);
    z_flat = fix_op->extend(z_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-truncate-and-extend");
    t_tmp = std::chrono::high_resolution_clock::now();

    // Scale
    _z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
        _z[i].assign((uint64_t*)z_offset, (uint64_t*)z_offset+new_hSize);
    }
}

void twoPartyGCNRelu(
    const ShareVecVec& _embedding, 
    ShareVecVec& _new_h, 
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    uint64_t taskNum = _embedding.size();
    size_t hSize = _embedding[0].size();

    std::vector<FixArray> z;
    z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        z[i] = fix_op->input(party, hSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
    }
    FixArray z_flat = concat(z);

    auto t_tmp = std::chrono::high_resolution_clock::now();

    FixArray new_h_flat = fix_op->relu(z_flat);
    new_h_flat = fix_op->extend(new_h_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNN-relu");

    _new_h.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        uint64_t* new_h_offset = (uint64_t*)new_h_flat.data + i*hSize;
        _new_h[i].assign((uint64_t*)new_h_offset, (uint64_t*)new_h_offset+hSize);
    } 
}

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
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    uint64_t taskNum = _embedding.size();
    size_t hSize = _embedding[0].size();
    size_t new_hSize = _weight[0].size();
    // std::vector<FixArray> weight;
    // weight.resize(hSize);
    // for (int i=0; i<hSize; ++i) {
    //     weight[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_weight[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     weight[i] = fix_op->reduce(weight[i], 32);
    // }

    // std::vector<FixArray> embedding(taskNum);
    // for (int i=0; i<taskNum; ++i) {
    //     embedding[i] = fix_op->input(party, hSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     embedding[i] = fix_op->reduce(embedding[i], 32);
    // }

    auto t_tmp = std::chrono::high_resolution_clock::now();

    ShareVecVec _z_reduce;
    twoPartyFHEMatMul(_embedding, _weight, _z_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNNPrediction-mat-mul");
    t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> z;
    z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        z[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_z_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray z_flat = concat(z);
    z_flat = fix_op->truncate_reduce(z_flat, SCALER_BIT_LENGTH);
    z_flat = fix_op->extend(z_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNNPrediction-truncate-and-extend");
    t_tmp = std::chrono::high_resolution_clock::now();

    // Scale
    if (_normalizer.size() != 0) {
        _z.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
            _z[i].assign((uint64_t*)z_offset, (uint64_t*)z_offset+new_hSize);
        }
        twoPartyGCNVectorScale(
            _z, 
            _normalizer, 
            _z, 
            true,
            dstTid, 
            party
        );
        for (int i=0; i<taskNum; ++i) {
            uint64_t* z_offset = (uint64_t*)z_flat.data + i*new_hSize;
            for (int j=0; j<new_hSize; ++j) *(z_offset+j) =  _z[i][j];
        }
    }

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNNPrediction-vector-scale");
    t_tmp = std::chrono::high_resolution_clock::now();

    FixArray z_flat_reduced = fix_op->reduce(z_flat, 32);
    z = deConcat(z_flat_reduced, taskNum, new_hSize);

    // std::vector<FixArray> z = fix_op->mul(embedding, weight, 32);
    std::vector<FixArray> new_h = fix_op->softmax(z, 32, SCALER_BIT_LENGTH);
    // FixArray z_flat = concat(z);
    // z_flat = fix_op->extend(z_flat, 64);
    FixArray new_h_flat = concat(new_h);
    new_h_flat = fix_op->extend(new_h_flat, 64);
    new_h = deConcat(new_h_flat, taskNum, new_hSize);
    z = deConcat(z_flat, taskNum, new_hSize);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNNPrediction-softmax");

    _z.resize(taskNum);
    _p.resize(taskNum);
    _p_minus_y.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        FixArray y = fix_op->input(party, new_hSize, (uint64_t*)&(_y[i][0]), true, 64, SCALER_BIT_LENGTH);
        FixArray p_minus_y = fix_op->sub(new_h[i], y);
        
        _z[i].assign((uint64_t*)z[i].data, (uint64_t*)z[i].data+new_hSize);
        _p[i].assign((uint64_t*)new_h[i].data, (uint64_t*)new_h[i].data+new_hSize);
        _p_minus_y[i].assign((uint64_t*)p_minus_y.data, (uint64_t*)p_minus_y.data+new_hSize);
    } 
}

void twoPartyGCNForwardNNPredictionWithoutWeight(
    const ShareVecVec& _embedding, 
    const ShareVecVec& _y,
    ShareVecVec& _p, 
    ShareVecVec& _p_minus_y, 
    int dstTid, 
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];

    uint64_t taskNum = _embedding.size();
    size_t hSize = _embedding[0].size();

    std::vector<FixArray> z;
    z.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        z[i] = fix_op->input(party, hSize, (uint64_t*)&(_embedding[i][0]), true, 64, SCALER_BIT_LENGTH);
    }
    FixArray z_flat = concat(z);
    FixArray z_flat_reduced = fix_op->reduce(z_flat, 32);
    z = deConcat(z_flat_reduced, taskNum, hSize);

    auto t_tmp = std::chrono::high_resolution_clock::now();

    // std::vector<FixArray> z = fix_op->mul(embedding, weight, 32);
    std::vector<FixArray> new_h = fix_op->softmax(z, 32, SCALER_BIT_LENGTH);
    // FixArray z_flat = concat(z);
    // z_flat = fix_op->extend(z_flat, 64);
    FixArray new_h_flat = concat(new_h);
    new_h_flat = fix_op->extend(new_h_flat, 64);
    new_h = deConcat(new_h_flat, taskNum, hSize);
    z = deConcat(z_flat, taskNum, hSize);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNForwardNNPrediction-softmax");

    _p.resize(taskNum);
    _p_minus_y.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        FixArray y = fix_op->input(party, hSize, (uint64_t*)&(_y[i][0]), true, 64, SCALER_BIT_LENGTH);
        FixArray p_minus_y = fix_op->sub(new_h[i], y);
        
        _p[i].assign((uint64_t*)new_h[i].data, (uint64_t*)new_h[i].data+hSize);
        _p_minus_y[i].assign((uint64_t*)p_minus_y.data, (uint64_t*)p_minus_y.data+hSize);
    } 
}

void twoPartyGCNBackwardNNInit(
    const ShareVecVec& _p_minus_y,
    const ShareVecVec& _ah_t,
    const ShareVecVec& _weight_t,
    const std::vector<uint64_t>& _normalizer,
    ShareVecVec& _d,
    ShareVecVec& _g,
    int dstTid,
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    
    uint64_t taskNum = _p_minus_y.size();
    size_t hSize = _p_minus_y[0].size();
    if (_weight_t.size() != hSize) {
        printf("Unmatched weight matrix and vector dimensions during ssGCNBackwardNNInit\n");
        exit(-1);
    }
    size_t new_hSize = _weight_t[0].size();
    // std::vector<FixArray> weight_t;
    // weight_t.resize(hSize);
    // for (int i=0; i<hSize; ++i) {
    //     weight_t[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_weight_t[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     weight_t[i] = fix_op->reduce(weight_t[i], 32);
    // }
    
    // std::vector<FixArray> p_minus_y(taskNum);
    // for (int i=0; i<taskNum; ++i) {
    //     p_minus_y[i] = fix_op->input(party, hSize, (uint64_t*)&(_p_minus_y[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     p_minus_y[i] = fix_op->reduce(p_minus_y[i], 32);
    // }

    // std::vector<FixArray> g = fix_op->mul(p_minus_y, weight_t, 32);

    auto t_tmp = std::chrono::high_resolution_clock::now();

    ShareVecVec _g_reduce;
    twoPartyFHEMatMul(_p_minus_y, _weight_t, _g_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNNInit-mat-mul-g");
    t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> g;
    g.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        g[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_g_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray g_flat = concat(g);
    g_flat = fix_op->truncate_reduce(g_flat, SCALER_BIT_LENGTH);
    g_flat = fix_op->extend(g_flat, 64);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNNInit-truncate-and-extend-g");
    t_tmp = std::chrono::high_resolution_clock::now();

    // FixArray g_flat = concat(g);
    // g_flat = fix_op->extend(g_flat, 64);
    g = deConcat(g_flat, taskNum, new_hSize);

    // std::vector<FixArray> ah_t(new_hSize);
    // for (int i=0; i<new_hSize; ++i) {
    //     ah_t[i] = fix_op->input(party, taskNum, (uint64_t*)&(_ah_t[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     ah_t[i] = fix_op->reduce(ah_t[i], 32);
    // }

    // std::vector<FixArray> d = fix_op->mul(ah_t, p_minus_y, 32); // TODO: get the mean
    // for (int i=0; i<new_hSize; ++i) {
    //     d[i] = fix_op->extend(d[i], 64);
    // }

    ShareVecVec _scaled_p_minus_y = _p_minus_y;
    if (_normalizer.size() != 0) {
        twoPartyGCNVectorScale(
            _p_minus_y, 
            _normalizer, 
            _scaled_p_minus_y, 
            true,
            dstTid, 
            party
        );
        if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNNInit-scale-p-minus-y");
        t_tmp = std::chrono::high_resolution_clock::now();
    }

    ShareVecVec _d_reduce;
    twoPartyFHEMatMul(_ah_t, _scaled_p_minus_y, _d_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNNInit-mat-mul-d");
    t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> d;
    d.resize(new_hSize);
    for (int i=0; i<new_hSize; ++i) {
        d[i] = fix_op->input(party, hSize, (uint64_t*)&(_d_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray d_flat = concat(d);
    d_flat = fix_op->truncate_reduce(d_flat, SCALER_BIT_LENGTH);
    d_flat = fix_op->extend(d_flat, 64);
    d = deConcat(d_flat, new_hSize, hSize);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNNInit-truncate-and-extend-d");
    
    _g.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        _g[i].assign((uint64_t*)g[i].data, (uint64_t*)g[i].data+new_hSize);
    }
    _d.resize(new_hSize);
    for (int j=0; j<new_hSize; ++j) {
        _d[j].assign((uint64_t*)d[j].data, (uint64_t*)d[j].data+hSize);
    }
}

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
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    uint64_t taskNum = _a_t_g.size();

    size_t hSize = _a_t_g[0].size();
    if (_weight_t.size() != hSize) {
        printf("Unmatched weight matrix and vector dimensions during ssGCNBackwardNNInit\n");
        exit(-1);
    }
    size_t new_hSize = _weight_t[0].size();
    // std::vector<FixArray> weight_t;
    // weight_t.resize(hSize);
    // for (int i=0; i<hSize; ++i) {
    //     weight_t[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_weight_t[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     weight_t[i] = fix_op->reduce(weight_t[i], 32);
    // }

    std::vector<FixArray> a_t_g(taskNum);
    std::vector<FixArray> z(taskNum);
    for (int i=0; i<taskNum; ++i) {
        a_t_g[i] = fix_op->input(party, hSize, (uint64_t*)&(_a_t_g[i][0]), true, 64, SCALER_BIT_LENGTH);
        z[i] = fix_op->input(party, hSize, (uint64_t*)&(_z[i][0]), true, 64, SCALER_BIT_LENGTH);
        // a_t_g[i] = fix_op->reduce(a_t_g[i], 32);
        // z[i] = fix_op->reduce(z[i], 32);
    }

    auto t_tmp = std::chrono::high_resolution_clock::now();

    FixArray a_t_g_flat = concat(a_t_g);
    FixArray z_flat = concat(z);
    BoolArray dz_flat = fix_op->drelu(z_flat);
    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-relu");
    t_tmp = std::chrono::high_resolution_clock::now();
    // FixArray dz_a_t_g_flat = fix_op->mul(dz_flat, a_t_g_flat, MultModBitLength);
    FixArray zero = fix_op->input(sci::ALICE, a_t_g_flat.size, (uint64_t)0, a_t_g_flat.signed_, a_t_g_flat.ell, a_t_g_flat.s);
    // FixArray dz_a_t_g_flat = fix_op->mul(dz_flat, a_t_g_flat, MultModBitLength);
    FixArray dz_a_t_g_flat = fix_op->if_else(dz_flat, a_t_g_flat, zero);
    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-elementwise-if-else");
    t_tmp = std::chrono::high_resolution_clock::now();
    // dz_a_t_g_flat = fix_op->truncate_reduce(dz_a_t_g_flat, SCALER_BIT_LENGTH);
    // dz_a_t_g_flat = fix_op->extend(dz_a_t_g_flat, 64);
    // if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-truncate-and-extend-dz_a_t_g_flat");
    // t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> dz_a_t_g = deConcat(dz_a_t_g_flat, taskNum, hSize);

    ShareVecVec _dz_a_t_g(taskNum);
    for (int i=0; i<taskNum; ++i) {
        _dz_a_t_g[i].assign((uint64_t*)dz_a_t_g[i].data, (uint64_t*)dz_a_t_g[i].data+hSize);
    }

    if (!isFirstLayer) {
        // std::vector<FixArray> g = fix_op->mul(dz_a_t_g, weight_t, 32);
        ShareVecVec _g_reduce;
        twoPartyFHEMatMul(_dz_a_t_g, _weight_t, _g_reduce, dstTid, party);

        std::vector<FixArray> g;
        g.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            g[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_g_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
        }
        FixArray g_flat = concat(g);
        g_flat = fix_op->truncate_reduce(g_flat, SCALER_BIT_LENGTH);
        g_flat = fix_op->extend(g_flat, 64);

        // FixArray g_flat = concat(g);
        // g_flat = fix_op->extend(g_flat, 64);
        g = deConcat(g_flat, taskNum, new_hSize);

        _g.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            _g[i].assign((uint64_t*)g[i].data, (uint64_t*)g[i].data+new_hSize);
        }
    }

    // std::vector<FixArray> ah_t(new_hSize);
    // for (int i=0; i<new_hSize; ++i) {
    //     ah_t[i] = fix_op->input(party, taskNum, (uint64_t*)&(_ah_t[i][0]), true, 64, SCALER_BIT_LENGTH);
    //     ah_t[i] = fix_op->reduce(ah_t[i], 32);
    // }

    // std::vector<FixArray> d = fix_op->mul(ah_t, dz_a_t_g, 32); // TODO: get the mean
    // for (int i=0; i<new_hSize; ++i) {
    //     d[i] = fix_op->extend(d[i], 64);
    // }

    ShareVecVec _scaled_dz_a_t_g = _dz_a_t_g;
    if (_normalizer.size() != 0) {
        twoPartyGCNVectorScale(
            _dz_a_t_g, 
            _normalizer, 
            _scaled_dz_a_t_g, 
            true,
            dstTid, 
            party
        );

        if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-scale-_dz_a_t_g");
        t_tmp = std::chrono::high_resolution_clock::now();
    }

    ShareVecVec _d_reduce;
    twoPartyFHEMatMul(_ah_t, _scaled_dz_a_t_g, _d_reduce, dstTid, party);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-mat-mul-d");
    t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> d;
    d.resize(new_hSize);
    for (int i=0; i<new_hSize; ++i) {
        d[i] = fix_op->input(party, hSize, (uint64_t*)&(_d_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
    }
    FixArray d_flat = concat(d);
    d_flat = fix_op->truncate_reduce(d_flat, SCALER_BIT_LENGTH);
    d_flat = fix_op->extend(d_flat, 64);
    d = deConcat(d_flat, new_hSize, hSize);

    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-truncate-and-extend-d");
    
    _d.resize(new_hSize);
    for (int j=0; j<new_hSize; ++j) {
        _d[j].assign((uint64_t*)d[j].data, (uint64_t*)d[j].data+hSize);
    }
}

void twoPartyGCNBackwardNNWithoutAH(
    const ShareVecVec& _a_t_g,
    const ShareVecVec& _z,
    const ShareVecVec& _weight_t,
    ShareVecVec& _dz_a_t_g,
    ShareVecVec& _g,
    bool isFirstLayer,
    int dstTid,
    int party
) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[dstTid]:BOB_fix_ops[dstTid];
    uint64_t taskNum = _a_t_g.size();

    size_t hSize = _a_t_g[0].size();
    if (_weight_t.size() != hSize) {
        printf("Unmatched weight matrix and vector dimensions during ssGCNBackwardNNInit\n");
        exit(-1);
    }
    size_t new_hSize = _weight_t[0].size();

    std::vector<FixArray> a_t_g(taskNum);
    std::vector<FixArray> z(taskNum);
    for (int i=0; i<taskNum; ++i) {
        a_t_g[i] = fix_op->input(party, hSize, (uint64_t*)&(_a_t_g[i][0]), true, 64, SCALER_BIT_LENGTH);
        z[i] = fix_op->input(party, hSize, (uint64_t*)&(_z[i][0]), true, 64, SCALER_BIT_LENGTH);
        // a_t_g[i] = fix_op->reduce(a_t_g[i], 32);
        // z[i] = fix_op->reduce(z[i], 32);
    }

    auto t_tmp = std::chrono::high_resolution_clock::now();

    FixArray a_t_g_flat = concat(a_t_g);
    FixArray z_flat = concat(z);
    BoolArray dz_flat = fix_op->drelu(z_flat);
    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-relu");
    t_tmp = std::chrono::high_resolution_clock::now();
    FixArray zero = fix_op->input(sci::ALICE, a_t_g_flat.size, (uint64_t)0, a_t_g_flat.signed_, a_t_g_flat.ell, a_t_g_flat.s);
    FixArray dz_a_t_g_flat = fix_op->if_else(dz_flat, a_t_g_flat, zero);
    if (party == sci::ALICE) print_duration(t_tmp, "twoPartyGCNBackwardNN-elementwise-if-else");
    t_tmp = std::chrono::high_resolution_clock::now();

    std::vector<FixArray> dz_a_t_g = deConcat(dz_a_t_g_flat, taskNum, hSize);

    _dz_a_t_g.resize(taskNum);
    for (int i=0; i<taskNum; ++i) {
        _dz_a_t_g[i].assign((uint64_t*)dz_a_t_g[i].data, (uint64_t*)dz_a_t_g[i].data+hSize);
    }

    if (!isFirstLayer) {
        // std::vector<FixArray> g = fix_op->mul(dz_a_t_g, weight_t, 32);
        ShareVecVec _g_reduce;
        twoPartyFHEMatMul(_dz_a_t_g, _weight_t, _g_reduce, dstTid, party);

        std::vector<FixArray> g;
        g.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            g[i] = fix_op->input(party, new_hSize, (uint64_t*)&(_g_reduce[i][0]), true, MultModBitLength, 2*SCALER_BIT_LENGTH);
        }
        FixArray g_flat = concat(g);
        g_flat = fix_op->truncate_reduce(g_flat, SCALER_BIT_LENGTH);
        g_flat = fix_op->extend(g_flat, 64);

        // FixArray g_flat = concat(g);
        // g_flat = fix_op->extend(g_flat, 64);
        g = deConcat(g_flat, taskNum, new_hSize);

        _g.resize(taskNum);
        for (int i=0; i<taskNum; ++i) {
            _g[i].assign((uint64_t*)g[i].data, (uint64_t*)g[i].data+new_hSize);
        }
    }
}

void twoPartyCmpSwap(std::vector<uint64_t>& lhs, std::vector<uint64_t>& rhs, int partyId, int party, int threadId) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[1 - partyId]:BOB_fix_ops[1 - partyId];

    size_t length = lhs.size();
    FixArray sci_lhs = fix_op->input(party, length, (uint64_t*)lhs.data(), false, 64);
    FixArray sci_rhs = fix_op->input(party, length, (uint64_t*)rhs.data(), false, 64);

    // Compute
    BoolArray selector;
    FixArray sum = fix_op->add(sci_lhs, sci_rhs);
    selector = fix_op->GT(sci_lhs, sci_rhs);
    FixArray result_lhs = fix_op->if_else(selector, sci_rhs, sci_lhs);
    FixArray result_rhs = fix_op->sub(sum, result_lhs);

    for (int i = 0; i < length; ++i) {
        lhs[i] = result_lhs.data[i];
        rhs[i] = result_rhs.data[i];
    }
}

// We compare the first element of each row of input, and swap the whole row.
void twoPartyCmpSwap(std::vector<std::vector<uint64_t>>& lhs, std::vector<std::vector<uint64_t>>& rhs, int partyId, int party, int threadId) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[1 - partyId]:BOB_fix_ops[1 - partyId];

    size_t num_cols = lhs[0].size();
    size_t length = lhs.size();
    size_t flat_length = length * num_cols;
    std::vector<uint64_t> cmp_lhs(length);
    std::vector<uint64_t> cmp_rhs(length);
    for (int i = 0; i < length; ++i) cmp_lhs[i] = lhs[i][0];
    for (int i = 0; i < length; ++i) cmp_rhs[i] = rhs[i][0];
    FixArray sci_cmp_lhs = fix_op->input(party, length, (uint64_t*)cmp_lhs.data(), false, 64);
    FixArray sci_cmp_rhs = fix_op->input(party, length, (uint64_t*)cmp_rhs.data(), false, 64);

    // Compute
    BoolArray selector;
    selector = fix_op->GT(sci_cmp_lhs, sci_cmp_rhs);
    BoolArray flat_selector(selector.party, flat_length);
    FixArray flat_lhs(party, flat_length, false, 64);
    FixArray flat_rhs(party, flat_length, false, 64);
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            flat_selector.data[i*num_cols + j] = selector.data[i];
            flat_lhs.data[i*num_cols + j] = lhs[i][j];
            flat_rhs.data[i*num_cols + j] = rhs[i][j];
        }
    }
    FixArray flat_sum = fix_op->add(flat_lhs, flat_rhs);
    FixArray result_flat_lhs = fix_op->if_else(flat_selector, flat_rhs, flat_lhs);
    FixArray result_flat_rhs = fix_op->sub(flat_sum, result_flat_lhs);

    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            lhs[i][j] = result_flat_lhs.data[i*num_cols + j];
            rhs[i][j] = result_flat_rhs.data[i*num_cols + j];
        }
    }
}

void twoPartyCmpOpen(std::vector<uint64_t>& lhs, std::vector<uint64_t>& rhs, std::vector<bool>& result, int partyId, int party, int threadId) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[1 - partyId]:BOB_fix_ops[1 - partyId];

    size_t length = lhs.size();
    FixArray sci_lhs = fix_op->input(party, length, (uint64_t*)lhs.data(), false, 64);
    FixArray sci_rhs = fix_op->input(party, length, (uint64_t*)rhs.data(), false, 64);

    // Compute
    FixArray sum = fix_op->add(sci_lhs, sci_rhs);
    BoolArray selector = fix_op->GT(sci_lhs, sci_rhs);
    BoolArray cmp_result = fix_op->bool_op->output(emp::PUBLIC, selector);
    
    result.resize(length);
    for (int i = 0; i < length; ++i) {
        // if (i <= (length >> 1)) result[i] = true;
        // else result[i] = false;
        result[i] = (cmp_result.data[i] == 1);
    }
}

void twoPartySelectedAssign(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, const std::vector<uint8_t>& selector, int partyId, int party, bool is_one_side, int threadId) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[1 - partyId]:BOB_fix_ops[1 - partyId];

    size_t length = dst.size();
    assert(src.size() == length);
    assert(selector.size() == length);
    FixArray sci_dst = fix_op->input(party, length, (uint64_t*)dst.data(), true, 64, SCALER_BIT_LENGTH);
    FixArray sci_src = fix_op->input(party, length, (uint64_t*)src.data(), true, 64, SCALER_BIT_LENGTH);
    BoolArray sci_selector = fix_op->bool_op->input(party, length, (uint8_t*)selector.data());

    FixArray sci_assigned;
    if (is_one_side) {
        sci_assigned = fix_op->one_side_if_else(sci_selector, sci_src, sci_dst);
    } else {
        sci_assigned = fix_op->if_else(sci_selector, sci_src, sci_dst);
    }

    FixArray sci_result = fix_op->output(emp::PUBLIC, sci_assigned);
    
    for (int i = 0; i < length; ++i) {
        dst[i] = sci_result.data[i];
    }
}

void twoPartyAdd(const std::vector<uint64_t>& lhs, const std::vector<uint64_t>& rhs, std::vector<uint64_t>& result, int partyId, int party, int threadId) {
    FixOp *fix_op = party==emp::ALICE? ALICE_fix_ops[1 - partyId]:BOB_fix_ops[1 - partyId];

    size_t length = lhs.size();
    FixArray sci_lhs = fix_op->input(party, length, (uint64_t*)lhs.data(), true, 64, SCALER_BIT_LENGTH);
    FixArray sci_rhs = fix_op->input(party, length, (uint64_t*)rhs.data(), true, 64, SCALER_BIT_LENGTH);
    // sci_lhs = fix_op->reduce(sci_lhs, 32);
    // sci_rhs = fix_op->reduce(sci_rhs, 32);

    // Compute
    FixArray sci_result = fix_op->add(sci_lhs, sci_rhs);
    sci_result = fix_op->extend(sci_result, 64);

    // printf("h1\n");
    sci_result = fix_op->output(emp::PUBLIC, sci_result);
    // printf("h2\n");
    
    result.resize(length);
    for (int i = 0; i < length; ++i) {
        result[i] = sci_result.data[i];
    }
}

void printShareVecVec(const ShareVecVec& svv, int dstTid, int party) {
    if (party == sci::ALICE) {
        TaskComm& taskComm = TaskComm::getClientInstance();
        taskComm.sendShareVecVec(svv, dstTid);
        ShareVecVec svv1;
        taskComm.recvShareVecVec(svv1, dstTid);
        DoubleTensor result;
        CryptoUtil::mergeShareAsDouble(result, svv, svv1);
        print_vector_of_vector(result, 10);
    } else {
        TaskComm& taskComm = TaskComm::getServerInstance();
        taskComm.sendShareVecVec(svv, dstTid);
        ShareVecVec svv1;
        taskComm.recvShareVecVec(svv1, dstTid);
        DoubleTensor result;
        CryptoUtil::mergeShareAsDouble(result, svv, svv1);
        // print_vector_of_vector(result);
    }
}

void getPlainShareVecVec(const ShareVecVec& svv, DoubleTensor& result, int dstTid, int party) {
    if (party == sci::ALICE) {
        TaskComm& taskComm = TaskComm::getClientInstance();
        ShareVecVec svv1;
        taskComm.recvShareVecVec(svv1, dstTid);
        CryptoUtil::mergeShareAsDouble(result, svv, svv1);        
    } else {
        TaskComm& taskComm = TaskComm::getServerInstance();
        taskComm.sendShareVecVec(svv, dstTid);
    }    
}

// Cross-entropy loss function
double cross_entropy_loss(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred) {
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

// Define the accuracy function (percentage of correct predictions)
double accuracy(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred) {
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
double accuracy(const std::vector<std::vector<double>>& Y, const std::vector<std::vector<double>>& Y_pred, const std::vector<bool>& is_border) {
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

// pack a vector of strings into a single string, and record the sizes of each packed string in sizev
std::string pack_string_vec(const std::vector<std::string>& sv, std::vector<size_t>& sizev) {
    std::string packed; // the packed string
    sizev.clear(); // clear the size vector
    for (const auto& s : sv) { // for each string in the vector
        packed += s; // append it to the packed string
        sizev.push_back(s.size()); // record its size in the size vector
    }
    return packed; // return the packed string
}

// unpack a packed string into a vector of strings, using the size vector to determine the boundaries
std::vector<std::string> unpack_string_vec(const std::string& packed, const std::vector<size_t>& sizev) {
    std::vector<std::string> sv; // the unpacked vector of strings
    size_t pos = 0; // the current position in the packed string
    for (const auto& size : sizev) { // for each size in the size vector
        sv.push_back(packed.substr(pos, size)); // extract the corresponding substring and push it to the vector
        pos += size; // update the position
    }
    return sv; // return the vector of strings
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

void setUpSCIChannel(int party, uint32_t dstTid) {
    TaskComm& taskComm = TaskComm::getClientInstance();
    bool isCluster = taskComm.getIsCluster();
    uint32_t mpcBasePort = taskComm.getMPCBasePort();
    uint32_t tileNum = taskComm.getTileNum();
    uint32_t tileIndex = taskComm.getTileIndex();
    TaskComm& serverTaskComm = TaskComm::getServerInstance();

    int port = party==emp::ALICE? (mpcBasePort + dstTid * tileNum + tileIndex) : (mpcBasePort + tileIndex * tileNum + dstTid);
    string address = "";
    if (!isCluster) {
        address = "127.0.0.1";
    } else {
        address = "10.0.0.";
        if (party==emp::ALICE) address += std::to_string(tileIndex + 1);
        else address += std::to_string(dstTid + 1);
    }

    if (party == emp::ALICE && !ALICE_iopacks[dstTid]) {
        ALICE_iopacks[dstTid] = new NetIO(party == emp::ALICE ? nullptr : address.c_str(), port);
        ALICE_otpacks[dstTid] = new OTPack<NetIO>(ALICE_iopacks[dstTid], party);
        ALICE_otpacks[dstTid]->setUpSilentOT(taskComm.getChannel(dstTid), taskComm.getOTRecverPtr(dstTid), taskComm.getOTSenderPtr(dstTid));
        ALICE_fix_ops[dstTid] = new FixOp(party, ALICE_iopacks[dstTid], ALICE_otpacks[dstTid]);
    } else if (party == emp::BOB && !BOB_iopacks[dstTid]) {
        BOB_iopacks[dstTid] = new NetIO(party == emp::ALICE ? nullptr : address.c_str(), port);
        BOB_otpacks[dstTid] = new OTPack<NetIO>(BOB_iopacks[dstTid], party);
        BOB_otpacks[dstTid]->setUpSilentOT(serverTaskComm.getChannel(dstTid), serverTaskComm.getOTRecverPtr(dstTid), serverTaskComm.getOTSenderPtr(dstTid));
        BOB_fix_ops[dstTid] = new FixOp(party, BOB_iopacks[dstTid], BOB_otpacks[dstTid]);
    }
}

void closeSCIChannel(int party, uint32_t dstTid) {
    if (party == emp::ALICE && ALICE_iopacks[dstTid]) {
        delete ALICE_iopacks[dstTid];
        ALICE_iopacks[dstTid] = nullptr;
        delete ALICE_otpacks[dstTid];
        ALICE_otpacks[dstTid] = nullptr;
        delete ALICE_fix_ops[dstTid];
        ALICE_fix_ops[dstTid] = nullptr;
    } else if (party == emp::BOB && BOB_iopacks[dstTid]) {
        delete BOB_iopacks[dstTid];
        BOB_iopacks[dstTid] = nullptr;
        delete BOB_otpacks[dstTid];
        BOB_otpacks[dstTid] = nullptr;
        delete BOB_fix_ops[dstTid];
        BOB_fix_ops[dstTid] = nullptr;
    }
}

void setUpSCIChannel() {
    // TaskComm& taskComm = TaskComm::getClientInstance();
    // bool isCluster = taskComm.getIsCluster();
    // uint32_t mpcBasePort = taskComm.getMPCBasePort();
    // uint32_t tileNum = taskComm.getTileNum();
    // uint32_t tileIndex = taskComm.getTileIndex();

    // ALICE_iopacks.resize(tileNum);
    // BOB_iopacks.resize(tileNum);
    // ALICE_otpacks.resize(tileNum);
    // BOB_otpacks.resize(tileNum);
    // ALICE_fix_ops.resize(tileNum);
    // BOB_fix_ops.resize(tileNum);
    // ALICE_fp_ops.resize(tileNum);
    // BOB_fp_ops.resize(tileNum);

    // std::vector<std::thread> threads;

    // for (uint32_t dstTid = 0; dstTid < tileNum; ++dstTid) {
    //     if (tileIndex != dstTid) {
    //         threads.emplace_back([mpcBasePort, dstTid, tileNum, tileIndex, isCluster](){
    //             uint32_t party = emp::ALICE;
    //             int port = party==emp::ALICE? (mpcBasePort + dstTid * tileNum + tileIndex) : (mpcBasePort + tileIndex * tileNum + dstTid);
    //             string address = "";
    //             if (!isCluster) {
    //                 address = "127.0.0.1";
    //             } else {
    //                 address = "10.0.0.";
    //                 if (party==emp::ALICE) address += std::to_string(tileIndex + 1);
    //                 else address += std::to_string(dstTid + 1);
    //             }
    //             ALICE_iopacks[dstTid] = new NetIO(party == emp::ALICE ? nullptr : address.c_str(), port);
    //             ALICE_otpacks[dstTid] = new OTPack<NetIO>(ALICE_iopacks[dstTid], party);
    //             ALICE_fix_ops[dstTid] = new FixOp(party, ALICE_iopacks[dstTid], ALICE_otpacks[dstTid]);
    //             ALICE_fp_ops[dstTid] = new FPOp(party, ALICE_iopacks[dstTid], ALICE_otpacks[dstTid]);
    //         });
    //     }
    // }

    // for (uint32_t dstTid = 0; dstTid < tileNum; ++dstTid) {
    //     if (tileIndex != dstTid) {
    //         threads.emplace_back([mpcBasePort, dstTid, tileNum, tileIndex, isCluster](){
    //             uint32_t party = emp::BOB;
    //             int port = party==emp::ALICE? (mpcBasePort + dstTid * tileNum + tileIndex) : (mpcBasePort + tileIndex * tileNum + dstTid);
    //             string address = "";
    //             if (!isCluster) {
    //                 address = "127.0.0.1";
    //             } else {
    //                 address = "10.0.0.";
    //                 if (party==emp::ALICE) address += std::to_string(tileIndex + 1);
    //                 else address += std::to_string(dstTid + 1);
    //             }
    //             BOB_iopacks[dstTid] = new NetIO(party == emp::ALICE ? nullptr : address.c_str(), port);
    //             BOB_otpacks[dstTid] = new OTPack<NetIO>(BOB_iopacks[dstTid], party);
    //             BOB_fix_ops[dstTid] = new FixOp(party, BOB_iopacks[dstTid], BOB_otpacks[dstTid]);
    //             BOB_fp_ops[dstTid] = new FPOp(party, BOB_iopacks[dstTid], BOB_otpacks[dstTid]);
    //         });
    //     }
    // }

    // for (auto& thrd : threads)
    //     thrd.join();

    TaskComm& taskComm = TaskComm::getClientInstance();
    uint32_t tileNum = taskComm.getTileNum();
    uint32_t tileIndex = taskComm.getTileIndex();

    ALICE_iopacks.resize(tileNum);
    BOB_iopacks.resize(tileNum);
    ALICE_otpacks.resize(tileNum);
    BOB_otpacks.resize(tileNum);
    ALICE_fix_ops.resize(tileNum);
    BOB_fix_ops.resize(tileNum);

    for (uint32_t dstTid = 0; dstTid < tileNum; ++dstTid) {
        if (tileIndex != dstTid) {
            ALICE_iopacks[dstTid] = nullptr;
            ALICE_otpacks[dstTid] = nullptr;
            ALICE_fix_ops[dstTid] = nullptr;

            BOB_iopacks[dstTid] = nullptr;
            BOB_otpacks[dstTid] = nullptr;
            BOB_fix_ops[dstTid] = nullptr;
        }
    }  
}

void closeSCIChannel() {
    // TaskComm& taskComm = TaskComm::getClientInstance();
    // uint32_t tileNum = taskComm.getTileNum();
    // uint32_t tileIndex = taskComm.getTileIndex();
    // for (uint32_t dstTid = 0; dstTid < tileNum; ++dstTid) {
    //     if (tileIndex != dstTid) {
    //         delete ALICE_iopacks[dstTid];
    //         delete ALICE_otpacks[dstTid];
    //         delete ALICE_fix_ops[dstTid];
    //         delete ALICE_fp_ops[dstTid];

    //         delete BOB_iopacks[dstTid];
    //         delete BOB_otpacks[dstTid];
    //         delete BOB_fix_ops[dstTid];
    //         delete BOB_fp_ops[dstTid];
    //     }
    // }    
}

} // namespace sci