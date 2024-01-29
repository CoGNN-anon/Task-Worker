#ifndef TASK_UTIL_H_
#define TASK_UTIL_H_

#include "emp-sh2pc/emp-sh2pc.h"

#include "ophelib/paillier_fast.h"
#include "ophelib/packing.h"
#include "ophelib/vector.h"
#include "ophelib/util.h"
#include "ophelib/ml.h"
#include "ophelib/integer.h"
#include "ophelib/wire.h"

#include "task.h"

#include "libOTe/Base/BaseOT.h"
#include "cryptoTools/Common/BitVector.h"
#include <libOTe/TwoChooseOne/SilentOtExtReceiver.h>
#include <libOTe/TwoChooseOne/SilentOtExtSender.h>

#include <random>
#include <unordered_map>
#include <cstdlib>
#include <vector>
#include <string>
#include <errno.h>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <type_traits>

// #include "LinearHE/utils-FHE.h"
#include "troy/app/TroyFHEWrapper.cuh"

#define SCALER_BIT_LENGTH 13

uint64_t intRand(const uint64_t & min = 0, const uint64_t & max = (1<<30));

class CryptoUtil {
public:
    CryptoUtil(CryptoUtil const&) = delete;
    CryptoUtil& operator=(CryptoUtil const&) = delete;

    static CryptoUtil& getInstance() {
        size_t k_size = 3072;
        static CryptoUtil theInstance(k_size);
        return theInstance;
    }

    void setUpPaillierCipher() {
        crypto.generate_keys();
        factor = SCALER_BIT_LENGTH;
    }

    void setUpFHECipher() {
        fhe_crypto = troyn::FHEWrapper();
    }

    void setUpRemotePaillierCipher(int tid, const ophelib::PublicKey& pk) {
        remote_crypto.emplace(tid, std::shared_ptr<ophelib::PaillierFast>(new ophelib::PaillierFast(pk)));
    }

    void setUpRemoteFHECipher(int tid, const string& buf, size_t pk_size, size_t gk_size, size_t autok_size) {
        remote_fhe_crypto.emplace(tid, std::shared_ptr<troyn::FHEWrapper>(new troyn::FHEWrapper(false)));
        remote_fhe_crypto.find(tid)->second->deserialize_pub_key(buf, pk_size, gk_size, autok_size);
    }

    void tileIndexIs(const uint32_t tid) {
        tileIndex_ = tid;
    }

    ophelib::Ciphertext encrypt(uint32_t plain) {
        return crypto.encrypt(ophelib::Integer(plain));
    }

    ophelib::Ciphertext encrypt(int32_t plain) {
        return crypto.encrypt(ophelib::Integer(plain));
    }

    ophelib::Ciphertext encrypt(uint64_t plain) {
        return crypto.encrypt(ophelib::Integer(plain));
    }

    ophelib::Ciphertext encrypt(double plain) {
        plain *= (1<<factor);
        ophelib::Integer n = (uint64_t)plain;
        return crypto.encrypt(n);
    }

    ophelib::Ciphertext encrypt(ophelib::Integer plain) {
        return crypto.encrypt(plain);
    }

    void serialize(const std::vector<uint64_t>& input, uint8_t* output, size_t& output_size) {
        output_size = input.size() * 8;
        for (size_t i = 0; i < input.size(); i++) {
            uint8_t array[8];
            memcpy(array, &input[i], sizeof(uint64_t));
            memcpy(output + i * 8, array, 8);
        }
    }

    void deserialize(const uint8_t* input, size_t input_size, std::vector<uint64_t>& output) {
        if (input_size % 8 != 0) {
            throw std::invalid_argument("Input size must be divisible by 8");
        }
        size_t output_size = input_size / 8;
        output.resize(output_size);
        for (size_t i = 0; i < output_size; i++) {
            uint8_t array[8];
            memcpy(array, input + i * 8, 8);
            memcpy(&output[i], array, sizeof(uint64_t));
        }
    }

    CipherEntry encryptToCipherEntry(uint32_t plain, const uint32_t tid) {
        CipherEntry ce;
        ophelib::Ciphertext cur_ct;
        if (tid == tileIndex_) {
            cur_ct = crypto.encrypt(ophelib::Integer(plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            cur_ct = cur_crypto->encrypt(ophelib::Integer(plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);         
        }
        ce.tid = tid;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
        ce.plainNum = 1;
        return ce;
    }

    CipherEntry encryptToCipherEntry(uint64_t plain, const uint32_t tid) {
        CipherEntry ce;
        ophelib::Ciphertext cur_ct;
        if (tid == tileIndex_) {
            cur_ct = crypto.encrypt(ophelib::Integer(plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            cur_ct = cur_crypto->encrypt(ophelib::Integer(plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);         
        }
        ce.tid = tid;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
        ce.plainNum = 1;
        return ce;
    }

    CipherEntry encryptToCipherEntry(std::vector<uint64_t>& plain, const uint32_t tid) {
        CipherEntry ce;
        ophelib::Ciphertext cur_ct;

        auto packed_plain = ophelib::Integer(0);
        const auto plaintext_bits = 64;
        auto pad = ophelib::Integer(1);
        pad = pad << plaintext_bits;
        for (auto x : plain) {
            packed_plain = packed_plain << (plaintext_bits + padding_bits);
            packed_plain = packed_plain + ophelib::Integer((uint64_t)x);
            packed_plain = packed_plain + pad;
        }

        // std::cout<< "Encrypt: " << packed_plain.to_string_() << std::endl;

        if (tid == tileIndex_) {
            cur_ct = crypto.encrypt(ophelib::Integer(packed_plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            cur_ct = cur_crypto->encrypt(ophelib::Integer(packed_plain));
            serializeCipherTextToBytes(cur_ct, ce.ct);         
        }
        ce.tid = tid;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
        ce.plainNum = plain.size();
        return ce;
    }

    CipherEntry encryptToCipherEntry_Mock(std::vector<uint64_t>& plain, const uint32_t tid) {
        CipherEntry ce;
        size_t output_size = 0;
        serialize(plain, (uint8_t*)ce.ct, output_size);
        if (output_size > MAX_CIPHER_TEXT_SIZE) {
            printf("FHE Cipher text size exceeds buffer during serialization! %lld!\n", output_size);
            exit(-1);
        }
        ce.tid = tid;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
        ce.plainNum = plain.size();
        return ce;
    }

    CipherEntry encryptToFHECipherEntry(std::vector<uint64_t>& plain, const uint32_t tid) {
        // seal::Ciphertext ct;
        std::stringstream os;
        CipherEntry ce;

        // if (tid == tileIndex_) {
        //     fhe_crypto.encrypt(ct, (std::vector<uint64_t>&)plain);
        // } else {
        //     const std::shared_ptr<FHEUtil>& cur_fhe_crypto = remote_fhe_crypto.find(tid)->second;
        //     cur_fhe_crypto->encrypt(ct, (std::vector<uint64_t>&)plain);        
        // }

        // ct.save(os);
        // uint64_t size = os.tellp();
        // if (size > MAX_CIPHER_TEXT_SIZE) {
        //     printf("FHE Cipher text size exceeds buffer during serialization! %lld!\n", size);
        //     exit(-1);
        // }
        // os.read((char*)ce.ct, os.tellp());

        // ce.tid = tid;
        // ce.isShare = false;
        // for (int i=0; i<2; ++i) ce.share[i] = 0;
        // ce.plainNum = plain.size();
        return ce;
    }

    std::string encryptToTroyFHECipherEntry(std::vector<uint64_t>& plain) {
        troyn::FHEWrapper& local_fhe = fhe_crypto;
        return local_fhe.encrypt(plain);
    }

    CipherEntry encryptToCipherEntry(int32_t plain, const uint32_t tid) {
        CipherEntry ce;
        ophelib::Ciphertext cur_ct;
        const ophelib::Integer int_plain = plain;
        if (tid == tileIndex_) {
            cur_ct = crypto.encrypt(int_plain);
            serializeCipherTextToBytes(cur_ct, ce.ct);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            cur_ct = cur_crypto->encrypt(int_plain);
            serializeCipherTextToBytes(cur_ct, ce.ct);         
        }
        ce.tid = tid;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
        ce.plainNum = 1;
        return ce;
    }

    CipherEntry encryptToCipherEntry(double plain, const uint32_t tid) {
        CipherEntry ce;
        plain *= (1<<factor);
        return encryptToCipherEntry((uint64_t)plain, tid);
    }

    void serializeCipherTextToCipherEntry(ophelib::Ciphertext& ct, CipherEntry& ce, const uint32_t tid, const uint32_t plainNum) {
        serializeCipherTextToBytes(ct, ce.ct);
        ce.tid = tid;
        ce.plainNum = plainNum;
        ce.isShare = false;
        for (int i=0; i<2; ++i) ce.share[i] = 0;
    }

    void serializeCipherTextToPackedCipherEntry(ophelib::Ciphertext& ct, PackedCipherEntry& pce, const uint32_t tid, const uint32_t plainNum) {
        serializeCipherTextToBytes(ct, pce.ct);
        pce.tid = tid;
        pce.plainNum = plainNum;
        pce.isShare = false;
    }

    void packCipherText(std::vector<CipherEntry*>& cev, std::vector<int32_t>& taskIdVec, std::vector<int32_t>& operandIdVec, std::vector<int32_t>& dstTidVec, int plain_num_per_ce, std::vector<std::shared_ptr<PackedCipherEntry>>& packed_ce_vec) {
        int ce_num = cev.size();
        int ce_num_per_packed = key_size / (plain_num_per_ce * (plaintext_bits + padding_bits));
        int packed_ce_num = ce_num / ce_num_per_packed;
        if (ce_num % ce_num_per_packed != 0) packed_ce_num += 1;
        packed_ce_vec.resize(packed_ce_num);

        #pragma omp parallel for
        for (int i = 0; i < packed_ce_num; ++i) {
            packed_ce_vec[i] = std::make_shared<PackedCipherEntry>();
            int cur_start = i * ce_num_per_packed;
            int cur_end = (ce_num < (i+1) * ce_num_per_packed) ? ce_num:(i+1) * ce_num_per_packed;
            
            ophelib::Ciphertext ct;
            deserializeCipherEntryToCipherText(ct, *(cev[cur_start]));
            packed_ce_vec[i]->taskId.push_back(taskIdVec[cur_start]);
            packed_ce_vec[i]->operandId.push_back(operandIdVec[cur_start]);
            packed_ce_vec[i]->dstTid.push_back(dstTidVec[cur_start]);

            ophelib::Integer scalar = ophelib::Integer(1);
            scalar = scalar << (plain_num_per_ce * (plaintext_bits + padding_bits));
            for (int j = cur_start+1; j < cur_end; ++j) {
                if (j % 10000 == 0) printf("pack share j = %d\n", j);
                ophelib::Ciphertext cur_ct;
                deserializeCipherEntryToCipherText(cur_ct, *(cev[j]));
                ct = ct * scalar;
                ct = ct + cur_ct;

                packed_ce_vec[i]->taskId.push_back(taskIdVec[j]);
                packed_ce_vec[i]->operandId.push_back(operandIdVec[j]);
                packed_ce_vec[i]->dstTid.push_back(dstTidVec[j]);
            }

            serializeCipherTextToPackedCipherEntry(ct, *packed_ce_vec[i], cev[cur_start]->tid,  (cur_end - cur_start) * plain_num_per_ce);
        }

        return;
    }

    // ophelib::Ciphertext encrypt(emp::Integer plain) {
    //     ophelib::Integer n = (uint64_t)plain;
    //     return crypto.encrypt(n);
    // }

    uint32_t decryptToUint32(ophelib::Ciphertext& ct) {
        return crypto.decrypt(ct).to_uint();
    }

    int32_t decryptToInt32(ophelib::Ciphertext& ct) {
        return crypto.decrypt(ct).to_int();
    }

    uint64_t decryptToUint64(ophelib::Ciphertext& ct) {
        return crypto.decrypt(ct).to_ulong();
    } 

    uint64_t decryptToInt64(ophelib::Ciphertext& ct) {
        return crypto.decrypt(ct).to_long();
    }

    void decryptToInt64Vec(ophelib::Ciphertext& ct, std::vector<uint64_t>& plains, int plainNum) {
        const auto plaintext_bits = 64;
        ophelib::Integer modulus = 1;
        modulus = modulus << plaintext_bits;
        ophelib::Integer packed_plain = crypto.decrypt(ct);
        packed_plain = packed_plain % (ophelib::Integer(1)<<((plaintext_bits + padding_bits) * plainNum));
        // std::cout<<"decrypt: "<<packed_plain<<std::endl;
        for (int i=plainNum-1; i>=0; --i) {
            ophelib::Integer cur_plain = (packed_plain >> ((plaintext_bits + padding_bits) * i)) % modulus;
            plains.push_back(cur_plain.to_ulong());
        }
        return;
    }

    uint64_t decryptCipherEntryToInt64(CipherEntry& ce) {
        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        return crypto.decrypt(ct).to_ulong();
    }

    void decryptCipherEntryToInt64Vec(CipherEntry& ce, std::vector<uint64_t>& plains) {
        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        decryptToInt64Vec(ct, plains, ce.plainNum);
        return;
    }

    void decryptCipherEntryToInt64Vec_Mock(CipherEntry& ce, std::vector<uint64_t>& plains) {
        deserialize((uint8_t*)ce.ct, ce.plainNum*8, plains);
        return;
    }

    void decryptFHECipherEntryToInt64Vec(CipherEntry& ce, std::vector<uint64_t>& plains) {
        // seal::Ciphertext ct;
        std::stringstream is;
        // is.write((char*)ce.ct, MAX_CIPHER_TEXT_SIZE);
        // ct.load(fhe_crypto.context, is);
        // fhe_crypto.decrypt(ct, (std::vector<uint64_t>&)plains);
        // // if (plains.size() != ce.plainNum) {
        // //     printf("Unequal FHE decrypted batch size! plains.size()=%d, ce.plainNum=%d\n", plains.size(), ce.plainNum);
        // //     exit(-1);
        // // }
        // plains.resize(ce.plainNum);
        return;
    }

    void decryptTroyFHECipherEntryToUint64Vec(const std::string& ce, size_t num_plains, std::vector<uint64_t>& plains) {
        troyn::FHEWrapper& local_fhe = fhe_crypto;
        plains = local_fhe.decrypt(ce, num_plains);
        return;
    }

    void decryptPackedCipherEntry(std::shared_ptr<PackedCipherEntry> pce) {
        ophelib::Ciphertext ct;
        deserializePackedCipherEntryToCipherText(ct, pce);
        decryptToInt64Vec(ct, pce->share, pce->plainNum);
        return;
    }

    double decryptToDouble(ophelib::Ciphertext& ct) {
        double result = (double)crypto.decrypt(ct).to_ulong();
        return result/(1<<factor);
    }

    // emp::Integer splitRandomEmpShareFromCipherText(ophelib::Ciphertext& ct, int tid) {
    //     uint32_t random_share = uni(rng);
    //     emp::Integer emp_share(64, random_share);
    //     ophelib::Integer he_share(random_share);
    //     ct -= (remote_crypto.find(tid)->second).encrypt(he_share);
    //     return emp_share;
    // }

    uint64_t getRandomShare() {
        uint64_t random_share = intRand();
        return (uint64_t)random_share;
    }

    uint64_t splitRandomShareFromCipherText(ophelib::Ciphertext& ct, int tid) {
        uint32_t random_share = intRand();
        ophelib::Integer he_share(random_share);
        const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
        ct -= cur_crypto->encrypt(he_share);
        return (uint64_t)random_share;
    }

    uint64_t splitRandomShareFromCipherEntry(CipherEntry& ce) {
        uint32_t random_share = intRand();
        const ophelib::Integer he_share = random_share;

        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        if (ce.tid == tileIndex_) {
            ct -= crypto.encrypt(he_share);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(ce.tid)->second;
            ophelib::Ciphertext share_ct = cur_crypto->encrypt(he_share);
            ct -= share_ct;
        }
        serializeCipherTextToBytes(ct, ce.ct, MAX_CIPHER_TEXT_SIZE);

        return (uint64_t)random_share;
    }


    void splitRandomShareFromCipherEntry(CipherEntry& ce, int shareNum, std::vector<uint64_t>& shares) {
        // Involve packing operation
        auto packed_he_share = ophelib::Integer(0);
        for (int i=0; i<shareNum; ++i) {
            packed_he_share = packed_he_share << (plaintext_bits + padding_bits);
            uint64_t random_share = intRand();
            shares.push_back(random_share);
            packed_he_share = packed_he_share + ophelib::Integer((uint64_t)random_share);
        }
        
        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        if (ce.tid == tileIndex_) {
            ct -= crypto.encrypt(packed_he_share);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(ce.tid)->second;
            ophelib::Ciphertext share_ct = cur_crypto->encrypt(packed_he_share);
            ct -= share_ct;
        }
        serializeCipherTextToBytes(ct, ce.ct, MAX_CIPHER_TEXT_SIZE);

        return;
    }

    void splitRandomShareFromCipherEntry_Mock(CipherEntry& ce, int shareNum, std::vector<uint64_t>& shares) {
        std::vector<uint64_t> values;
        deserialize((uint8_t*)ce.ct, shareNum * 8, values);
        for (int i = 0; i < shareNum; ++i) {
            uint64_t random_share = intRand();
            shares.push_back(random_share);
            values[i] -= shares[i];
        }
        size_t output_size = 0;
        serialize(values, (uint8_t*)ce.ct, output_size);
        if (output_size > MAX_CIPHER_TEXT_SIZE) {
            printf("Cipher text size exceeds buffer during serialization! %lld!\n", output_size);
            exit(-1);
        }

        return;
    }

    void splitRandomShareFromFHECipherEntry(CipherEntry& ce, int shareNum, std::vector<uint64_t>& shares) {
        for (int i=0; i<shareNum; ++i) {
            uint64_t random_share = intRand();
            shares.push_back(random_share);
        }

        // const std::shared_ptr<FHEUtil>& cur_fhe_crypto = remote_fhe_crypto.find(ce.tid)->second;
        // seal::Plaintext packed_pt;
        // // printf("shares size %d\n", shares.size());
        // // printf("ce.plainNum %d\n", ce.plainNum);
        // // printf("ce.tid %d\n", ce.tid);
        // // printf("cur_fhe_crypto %lld\n", cur_fhe_crypto);
        // // printf("encoder %lld\n", cur_fhe_crypto->encoder);
        // cur_fhe_crypto->encoder->encode((std::vector<uint64_t>&)shares, packed_pt);
        
        // seal::Ciphertext ct;
        // std::stringstream is;
        // is.write((char*)ce.ct, MAX_CIPHER_TEXT_SIZE);
        // ct.load(fhe_crypto.context, is);

        // cur_fhe_crypto->evaluator->sub_plain_inplace(ct, packed_pt);

        // std::stringstream os;
        // ct.save(os);
        // uint64_t size = os.tellp();
        // if (size > MAX_CIPHER_TEXT_SIZE) {
        //     printf("FHE Cipher text size exceeds buffer during serialization! %lld!\n", size);
        //     exit(-1);
        // }
        // memset(ce.ct, 0, MAX_CIPHER_TEXT_SIZE);
        // os.read((char*)ce.ct, os.tellp());

        return;
    }

    void splitRandomShareFromTroyFHECipherEntry(std::string& ce, uint32_t tid, int shareNum, std::vector<uint64_t>& shares) {
        troyn::FHEWrapper& remote_fhe = *(remote_fhe_crypto.find(tid)->second);
        shares = remote_fhe.subtract_random(ce, shareNum);
        return;
    }

    void splitRandomShareFromPackedCipherEntry(std::shared_ptr<PackedCipherEntry> pce) {
        // Involve packing operation
        auto packed_he_share = ophelib::Integer(0);
        pce->share.clear();
        for (int i=0; i<pce->plainNum; ++i) {
            packed_he_share = packed_he_share << (plaintext_bits + padding_bits);
            uint64_t random_share = intRand();
            pce->share.push_back(random_share);
            packed_he_share = packed_he_share + ophelib::Integer((uint64_t)random_share);
        }
        
        ophelib::Ciphertext ct;
        deserializePackedCipherEntryToCipherText(ct, pce);
        if (pce->tid == tileIndex_) {
            ct -= crypto.encrypt(packed_he_share);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(pce->tid)->second;
            ophelib::Ciphertext share_ct = cur_crypto->encrypt(packed_he_share);
            ct -= share_ct;
        }
        serializeCipherTextToBytes(ct, pce->ct, MAX_CIPHER_TEXT_SIZE);

        return;
    }

    void mergeShareIntoCipherText(ophelib::Ciphertext& ct, uint64_t share, int tid) {
        ophelib::Integer he_share(share);
        const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
        ct = ct + cur_crypto->encrypt(he_share);
        return;
    }

    void mergeShareIntoCipherEntry(CipherEntry& ce, uint64_t share, int tid) {
        ophelib::Integer he_share(share);

        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
        ct = ct + cur_crypto->encrypt(he_share);
        serializeCipherTextToBytes(ct, ce.ct, MAX_CIPHER_TEXT_SIZE);

        return;
    }

    void mergeShareIntoCipherEntry(CipherEntry& ce, std::vector<uint64_t>& shares, int tid) {
        auto packed_share = ophelib::Integer(0);
        const auto plaintext_bits = 64;
        for (auto x : shares) {
            packed_share = packed_share << (plaintext_bits + padding_bits);
            packed_share = packed_share + ophelib::Integer((uint64_t)x);
        }

        ophelib::Ciphertext ct;
        deserializeCipherEntryToCipherText(ct, ce);
        if (ce.tid == tileIndex_) {
            ct = ct + crypto.encrypt(packed_share);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            ct = ct + cur_crypto->encrypt(packed_share);
        }
        serializeCipherTextToBytes(ct, ce.ct, MAX_CIPHER_TEXT_SIZE);

        return;
    }

    static void intoShares(uint32_t input, uint64_t& share0, uint64_t& share1) {
        share0 = intRand();
        share1 = ((uint64_t)input) - share0;
    }

    static uint32_t mergeShareAsUint32(uint64_t share0, uint64_t share1) {
        return (uint32_t)(share0 + share1);
    }

    static void intoShares(uint64_t input, uint64_t& share0, uint64_t& share1) {
        share0 = intRand();
        share1 = input - share0;
    }

    static uint64_t mergeShareAsUint64(uint64_t share0, uint64_t share1) {
        return (uint64_t)(share0 + share1);
    }

    template<typename PlainType>
    PlainType mergeShareAs(uint64_t share0, uint64_t share1) {
        if (std::is_floating_point<PlainType>::value) return ((PlainType)(share0 + share1))/(1<<factor);
        else return (PlainType)(share0 + share1);
    }

    template<typename PlainType>
    PlainType decodeFixedPointAs(uint64_t encoded) {
        if (std::is_floating_point<PlainType>::value) return ((PlainType)(encoded))/(1<<factor);
        else return (PlainType)(encoded);
    }

    void mergeShareVec(std::vector<uint64_t>& shareVec0, std::vector<uint64_t>& shareVec1) {
        ophelib::Integer packed_share0 = shareVecToOPHEInteger(shareVec0);
        ophelib::Integer packed_share1 = shareVecToOPHEInteger(shareVec1);
        packed_share0 = packed_share0 + packed_share1;
        int num = shareVec0.size();
        shareVec0.clear();
        OPHEIntegerToInt64Vec(shareVec0, packed_share0, num);
        return;
    }

    static void intoShares(double input, uint64_t& share0, uint64_t& share1) {
        int64_t scaled = input * (1<<SCALER_BIT_LENGTH);
        share0 = intRand();
        share1 = scaled - share0;
    }

    static int64_t extend(uint64_t x) {
        // Mask the lower 44 bits of x
        uint64_t mask = (1ULL << 44) - 1;
        uint64_t lower = x & mask;

        // Check if the sign bit is 1
        bool negative = (lower >> 43) & 1;

        // If negative, extend the sign to the upper 20 bits of x
        if (negative) {
            uint64_t sign = ~mask;
            x = x | sign;
        }

        // Cast x to int64_t
        return static_cast<int64_t>(x);
    }

    static double mergeShareAsDouble(uint64_t share0, uint64_t share1) {
        // printf("%lf\n", (double)((share0 + share1)/(1<<SCALER_BIT_LENGTH)));
        uint64_t sum_share = (share0 + share1) % (1ul << 44);
        return ((double)((int64_t)(extend(sum_share))))/(1<<SCALER_BIT_LENGTH);
    }

    static void intoShares(const DoubleTensor& input, ShareTensor& share0, ShareTensor& share1) {
        share0.resize(input.size());
        share1.resize(input.size());
        for (int i = 0; i < input.size(); ++i) {
            share0[i].resize(input[i].size());
            share1[i].resize(input[i].size());
            for (int j = 0; j < input[i].size(); ++j) intoShares(input[i][j], share0[i][j], share1[i][j]);            
        }
    }

    static void intoShares(const std::vector<double>& input, std::vector<uint64_t>& share0, std::vector<uint64_t>& share1) {
        share0.resize(input.size());
        share1.resize(input.size());
        for (int i = 0; i < input.size(); ++i) {
            intoShares(input[i], share0[i], share1[i]);            
        }
    }

    static uint64_t encodeDoubleAsFixedPoint(double input) {
        return (uint64_t)(input*(1<<SCALER_BIT_LENGTH));
    }

    static void mergeShareAsDouble(DoubleTensor& output, const ShareTensor& share0, const ShareTensor& share1) {
        output.resize(share0.size());
        for (int i = 0; i < output.size(); ++i) {
            output[i].resize(share0[i].size());
            for (int j = 0; j < output[i].size(); ++j) output[i][j] = mergeShareAsDouble(share0[i][j], share1[i][j]);            
        }
    }

    static void mergeShareAsDouble(std::vector<double>& output, const std::vector<uint64_t>& share0, const std::vector<uint64_t>& share1) {
        output.resize(share0.size());
        for (int i = 0; i < output.size(); ++i) {
            output[i] = mergeShareAsDouble(share0[i], share1[i]);            
        }
    }

    void addShareVec(std::vector<uint64_t>& resultShareVec, const std::vector<uint64_t>& shareVec0, const std::vector<uint64_t>& shareVec1) {
        size_t shareVecSize = shareVec0.size();
        if (shareVecSize != shareVec1.size() || shareVecSize != resultShareVec.size()) {
            printf("Unequal shareVecSize to be added! %d %d %d\n", resultShareVec.size(), shareVec0.size(), shareVec1.size());
            exit(-1);
        }
        for (int i=0; i<shareVecSize; ++i) {
            resultShareVec[i] = shareVec0[i] + shareVec1[i];
        }
    }

    void substractShareVec(std::vector<uint64_t>& resultShareVec, const std::vector<uint64_t>& shareVec0, const std::vector<uint64_t>& shareVec1) {
        size_t shareVecSize = shareVec0.size();
        if (shareVecSize != shareVec1.size() || shareVecSize != resultShareVec.size()) {
            printf("Unequal shareVecSize to be substracted! %d %d %d\n", resultShareVec.size(), shareVec0.size(), shareVec1.size());
            exit(-1);
        }
        for (int i=0; i<shareVecSize; ++i) {
            resultShareVec[i] = shareVec0[i] - shareVec1[i];
        }
    }

    emp::Integer shareToEmpInteger(uint64_t share, int party) {
        return emp::Integer(64, share, party);
    }

    emp::Integer doubleToEmpInteger(double value, int party) {
        return emp::Integer(64, (uint64_t)(value*(1<<factor)), party);
    }

    emp::Integer uint32ToEmpInteger(uint32_t value, int party) {
        return emp::Integer(64, (uint64_t)value, party);
    }

    emp::Integer shareVecToEmpInteger(std::vector<uint64_t> shareVec, int party) {
        int shareNum = shareVec.size();
        emp::Integer result(64*shareNum, 0, emp::PUBLIC);
        for (int i=0; i<shareNum; ++i) {
            result = result << 64;
            // result = result + emp::Integer(64*shareNum, shareVec[i], party);
            result = result | (emp::Integer(64, shareVec[i], party).resize(64*shareNum, false));
        }
        return result;
    }

    ophelib::Integer shareVecToOPHEInteger(std::vector<uint64_t> shareVec) {
        auto packed_share = ophelib::Integer(0);
        const auto plaintext_bits = 64;
        for (auto x : shareVec) {
            packed_share = packed_share << plaintext_bits;
            packed_share = packed_share + ophelib::Integer((uint64_t)x);
        }
        return packed_share;
    }

    void OPHEIntegerToInt64Vec(std::vector<uint64_t>& shareVec, ophelib::Integer& packed, int num) {
        const auto plaintext_bits = 64;
        ophelib::Integer modulus = 1;
        modulus = modulus << plaintext_bits;
        packed = packed % (ophelib::Integer(1)<<(plaintext_bits * num));
        // std::cout<<"Packed plain: "<<packed.to_string_()<<std::endl;
        for (int i=num-1; i>=0; --i) {
            ophelib::Integer cur = (packed >> (plaintext_bits * i)) % modulus;
            shareVec.push_back(cur.to_ulong());
        }
    }

    emp::Float shareToEmpFloat(uint64_t share, int party) {
        double float_share = ((double)share)/(1<<factor);
        return emp::Float(share, party);
    }

    uint64_t empFloatToShare(emp::Float f, int party) {
        return (uint64_t)((f.reveal<double>(party))*(1<<factor));
    }

    uint64_t empIntegerToShare(emp::Integer n, int party) {
        return (uint64_t)(n.reveal<uint64_t>(party));
    }

    size_t serializeCipherTextToBytes(const ophelib::Ciphertext& ct, uint8_t* buf, size_t size_limit=MAX_CIPHER_TEXT_SIZE) {
        memset(buf, 0, size_limit);

        flatbuffers::FlatBufferBuilder builder;
        auto x = ophelib::serialize(builder, ct);
        builder.Finish(x);

        uint8_t* srcBuf = builder.GetBufferPointer();
        size_t size = builder.GetSize();

        if (size > size_limit) {
            printf("Cipher text size exceeds buffer during serialization! %lld!\n", size);
            exit(-1);
        }
        memcpy(buf, srcBuf, size);

        return size;
    }

    void deserializeBytesToCipherText(ophelib::Ciphertext& ct, uint8_t* buf, const std::shared_ptr<ophelib::Integer>& n2_shared, const std::shared_ptr<ophelib::FastMod>& fast_mod) {
        ophelib::deserialize(ophelib::Wire::GetCiphertext(buf)->data(), ct.data);
        ct.n2_shared = n2_shared;
        ct.fast_mod = fast_mod;
    }

    void deserializeCipherEntryToCipherText(ophelib::Ciphertext& ct, CipherEntry& ce) {
        if (ce.tid == tileIndex_) {
            const std::shared_ptr<ophelib::Integer>& n2_shared = crypto.get_n2();
            const std::shared_ptr<ophelib::FastMod>& fast_mod = crypto.get_fast_mod();
            deserializeBytesToCipherText(ct, ce.ct, n2_shared, fast_mod);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(ce.tid)->second;
            const std::shared_ptr<ophelib::Integer>& n2_shared = cur_crypto->get_n2();
            const std::shared_ptr<ophelib::FastMod>& fast_mod = cur_crypto->get_fast_mod();
            deserializeBytesToCipherText(ct, ce.ct, n2_shared, fast_mod);
        }
    }

    void deserializePackedCipherEntryToCipherText(ophelib::Ciphertext& ct, std::shared_ptr<PackedCipherEntry> pce) {
        if (pce->tid == tileIndex_) {
            const std::shared_ptr<ophelib::Integer>& n2_shared = crypto.get_n2();
            const std::shared_ptr<ophelib::FastMod>& fast_mod = crypto.get_fast_mod();
            deserializeBytesToCipherText(ct, pce->ct, n2_shared, fast_mod);
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(pce->tid)->second;
            const std::shared_ptr<ophelib::Integer>& n2_shared = cur_crypto->get_n2();
            const std::shared_ptr<ophelib::FastMod>& fast_mod = cur_crypto->get_fast_mod();
            deserializeBytesToCipherText(ct, pce->ct, n2_shared, fast_mod);
        }
    }

    // void deserializeBytesToPackedCipherText(ophelib::PackedCiphertext& pct, uint8_t* buf, const std::shared_ptr<ophelib::Integer>& n2_shared, const std::shared_ptr<ophelib::FastMod>& fast_mod) {
    //     ophelib::deserialize(buf, pct);
    //     pct.data.n2_shared = n2_shared;
    //     pct.data.fast_mod = fast_mod;
    // }

    // void deserializeCipherEntryToPackedCipherText(ophelib::PackedCiphertext& pct, CipherEntry& ce) {
    //     if (ce.tid == tileIndex_) {
    //         const std::shared_ptr<ophelib::Integer>& n2_shared = crypto.get_n2();
    //         const std::shared_ptr<ophelib::FastMod>& fast_mod = crypto.get_fast_mod();
    //         deserializeBytesToPackedCipherText(pct, ce.ct, n2_shared, fast_mod);
    //     } else {
    //         const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(ce.tid)->second;
    //         const std::shared_ptr<ophelib::Integer>& n2_shared = cur_crypto->get_n2();
    //         const std::shared_ptr<ophelib::FastMod>& fast_mod = cur_crypto->get_fast_mod();
    //         deserializeBytesToPackedCipherText(pct, ce.ct, n2_shared, fast_mod);
    //     }        
    // }

    size_t serializePublicKeyToBytes(const ophelib::PublicKey& pk, uint8_t* buf, size_t size_limit=MAX_PUBLIC_KEY_SIZE) {
        memset(buf, 0, size_limit);

        flatbuffers::FlatBufferBuilder builder;
        auto x = ophelib::serialize(builder, pk);
        builder.Finish(x);

        uint8_t* srcBuf = builder.GetBufferPointer();
        size_t size = builder.GetSize();

        if (size > size_limit) {
            printf("Public key size exceeds buffer during serialization! %lld!\n", size);
        }
        memcpy(buf, srcBuf, size);

        return size;
    }

    void deserializeBytesToPublicKey(ophelib::PublicKey& pk, uint8_t* buf) {
        ophelib::deserialize(buf, pk);
    }

    const ophelib::PublicKey& getPublicKey(int tid) {
        if (tid == tileIndex_) {
            return crypto.get_pub();
        } else {
            const std::shared_ptr<ophelib::PaillierFast>& cur_crypto = remote_crypto.find(tid)->second;
            return cur_crypto->get_pub();
        }
    }

    const ophelib::PublicKey& getPublicKey() {
        return crypto.get_pub();
    }

    troyn::FHEWrapper& getFHEPublicKey() {
        return fhe_crypto;
    }

    int32_t getPlainTextBits() {
        return plaintext_bits;
    }

    int32_t getPaddingBits() {
        return padding_bits;
    }

    uint32_t getKeySize() {
        return key_size;
    }    

    ophelib::PaillierFast crypto;

    std::unordered_map<int, std::shared_ptr<troyn::FHEWrapper>> remote_fhe_crypto;
    troyn::FHEWrapper fhe_crypto;
    std::mutex fhe_mut;

private:
    // CryptoUtil (size_t key_size): crypto(key_size), rng((std::random_device())()), uni(0, (1<<25)) {}
    CryptoUtil (size_t k_size): crypto(k_size), key_size(k_size), plaintext_bits(64), padding_bits(2) {}
    std::unordered_map<int, std::shared_ptr<ophelib::PaillierFast>> remote_crypto;
    uint32_t key_size;
    uint32_t factor = SCALER_BIT_LENGTH;
    // std::mt19937 rng;
    // std::uniform_int_distribution<uint32_t> uni;
    uint32_t tileIndex_;
    const int32_t plaintext_bits;
    const int32_t padding_bits;
};

class Semaphore
{
private:
    size_t avail;
    std::mutex m;
    std::condition_variable cv;

public:
    // only one thread can call this; by default, we construct a binary semaphore
    explicit Semaphore(int avail_ = 0) : avail(avail_) {}

    void acquire() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this] { return avail > 0; });
        avail--;
        cv.notify_one();
    }

    void release() {
        std::unique_lock<std::mutex> lk(m);
        avail++;
        cv.notify_one();
    }
};

#endif