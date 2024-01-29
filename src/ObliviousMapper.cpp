#include "ObliviousMapper.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/access.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

#define MY_THREAD_NUM 48
#define FHE_SLOT_NUM 8192

void client_oblivious_mapper_preprocess(std::vector<uint64_t>& srcPos, std::vector<uint64_t>& dstPos, 
                                    uint64_t plainNumPerPos, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid, bool allowMissing) {
    uint64_t srcPosNum = srcPos.size();
    uint64_t dstPosNum = dstPos.size();
	TaskComm& taskComm = TaskComm::getClientInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    // Build mapping from srcPos value to srcPos index
    std::unordered_map<uint64_t, uint64_t> posMap;
    for (int i=0; i<srcPosNum; ++i) {
        // Insert the first index of a value only
        if (posMap.find(srcPos[i]) == posMap.end())
            posMap.insert({srcPos[i], i});
    }

    taskComm.sendObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);
    
    // Initialize s (random ShareVecVec of length dstPosNum)
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    // Recv srcCev
    CipherEntryVec srcCev;
    taskComm.recvCipherEntryVec(srcCev, dstTid);
    // Do mapping & random share splitting
    CipherEntryVec dstCev;
    dstCev.ceVecs.resize(dstPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing client split random share (s) i=%d\n", i); 
        auto search = posMap.find(dstPos[i]);
        uint64_t srcIndex = 0;
        if (search == posMap.end()) {
            if (!allowMissing) {
                printf("Did not find pos index in posMap! %lld\n", dstPos[i]);
                exit(-1);
            }
        } else {
            srcIndex = search->second;
        }
        dstCev.ceVecs[i] = srcCev.ceVecs[srcIndex];
#ifdef USE_FHE
        cryptoUtil.splitRandomShareFromFHECipherEntry(dstCev.ceVecs[i], plainNumPerPos, dstSvv[i]);
#else
        cryptoUtil.splitRandomShareFromCipherEntry(dstCev.ceVecs[i], plainNumPerPos, dstSvv[i]);
#endif
    }

    // Send dstCev
    taskComm.sendCipherEntryVec(dstCev, dstTid);   

    // Store dstSvv (s)
    std::string fileName = "client_dstSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    writeShareVecVectoFile(dstSvv, fileName);
}

void server_oblivious_mapper_preprocess(uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    uint64_t srcPosNum = 0;
    uint64_t dstPosNum = 0;
    uint64_t plainNumPerPos = 0;
	TaskComm& taskComm = TaskComm::getServerInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    taskComm.recvObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);

    // Generate r (random ShareVecVec of length srcPosNum)
    ShareVecVec srcSvv;
    srcSvv.resize(srcPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        srcSvv[i].resize(plainNumPerPos);
        for (int j=0; j<plainNumPerPos; ++j) {
            srcSvv[i][j] = cryptoUtil.getRandomShare();
        }
    }

    // Encrypt srcSvv
    CipherEntryVec srcCev;
    srcCev.ceVecs.resize(srcPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server encrypt random share ([r]) i=%d\n", i); 
        // srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry(srcSvv[i], tileIndex);
#ifdef USE_FHE
        srcCev.ceVecs[i] = cryptoUtil.encryptToFHECipherEntry(srcSvv[i], tileIndex);
#else
        srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry(srcSvv[i], tileIndex);
#endif
    }    

    // Send srcCev
    taskComm.sendCipherEntryVec(srcCev, dstTid);

    // Recv dstCev
    CipherEntryVec dstCev;
    taskComm.recvCipherEntryVec(dstCev, dstTid);
    if (dstCev.ceVecs.size() != dstPosNum) {
        printf("Illegal received dstCev size!\n");
        exit(-1);
    }

    // Do decryption
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)  
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server decrypt random share (M*r-s) i=%d\n", i); 
        // cryptoUtil.decryptCipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
#ifdef USE_FHE
        cryptoUtil.decryptFHECipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
#else
        cryptoUtil.decryptCipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
#endif
    }

    // Store dstSvv (M*r-s)
    std::string fileName = "server_dstSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    writeShareVecVectoFile(dstSvv, fileName);

    // Store srcSvv (r)
    fileName = "server_srcSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    writeShareVecVectoFile(srcSvv, fileName);
}

uint64_t client_batch_oblivious_mapper_preprocess(std::vector<uint64_t>& srcPos, std::vector<uint64_t>& dstPos, 
                                    uint64_t plainNumPerPos, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid, bool allowMissing) {
    uint64_t srcPosNum = srcPos.size();
    uint64_t dstPosNum = dstPos.size();
	TaskComm& taskComm = TaskComm::getClientInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    uint32_t keySize = cryptoUtil.getKeySize();
    uint32_t plainTextBits = cryptoUtil.getPlainTextBits();
    uint32_t paddingBits = cryptoUtil.getPaddingBits();
    // uint32_t batchSize = keySize / ((plainTextBits + paddingBits) * plainNumPerPos); // Max pos num per cipher entry
#ifdef USE_FHE 
    uint32_t batchSize = FHE_SLOT_NUM / plainNumPerPos;
#else
    uint32_t batchSize = 1;
#endif
    uint32_t plainNumPerCipherEntry = plainNumPerPos * batchSize;

    // Build mapping from srcPos value to srcPos index
    std::unordered_map<uint64_t, uint64_t> posMap;
    for (int i=0; i<srcPosNum; ++i) {
        // Insert the first index of a value only
        if (posMap.find(srcPos[i]) == posMap.end())
            posMap.insert({srcPos[i], i});
    }

    taskComm.sendObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);
    
    // Initialize s (random ShareVecVec of length dstPosNum)
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    // Recv srcCev
    CipherEntryVec srcCev;
    taskComm.recvCipherEntryVec(srcCev, dstTid);
    // Do mapping & random share splitting
    CipherEntryVec dstCev;
    dstCev.ceVecs.resize(dstPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing client split random share (s) i=%d\n", i); 
        auto search = posMap.find(dstPos[i]);
        uint64_t srcIndex = 0;
        if (search == posMap.end()) {
            if (!allowMissing) {
                printf("Did not find pos index in posMap! %lld\n", dstPos[i]);
                exit(-1);
            }
        } else {
            srcIndex = search->second;
        }
        dstCev.ceVecs[i] = srcCev.ceVecs[srcIndex];
        // printf("party %d client, srcCev.ceVecs[srcIndex].tid %d dstCev.ceVecs[i].tid %d\n", tileIndex, srcCev.ceVecs[srcIndex].tid, dstCev.ceVecs[i].tid);
#ifdef USE_FHE
        cryptoUtil.splitRandomShareFromFHECipherEntry(dstCev.ceVecs[i], plainNumPerCipherEntry, dstSvv[i]);
#else
        // cryptoUtil.splitRandomShareFromCipherEntry(dstCev.ceVecs[i], plainNumPerCipherEntry, dstSvv[i]);
        cryptoUtil.splitRandomShareFromCipherEntry_Mock(dstCev.ceVecs[i], plainNumPerCipherEntry, dstSvv[i]);
#endif
    }

    // Send dstCev
    taskComm.sendCipherEntryVec(dstCev, dstTid);   

    std::cout<<"client dstSvv[0].size() = "<<dstSvv[0].size()<<std::endl;

    for (int i=0; i<batchSize; ++i) {
        // Store dstSvv (s)
        std::string fileName = "client_dstSvv_";
        fileName = fileName + std::to_string(iter + i) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
        ShareVecVec curDstSvv;
        curDstSvv.resize(dstPosNum);
        for (int j=0; j<dstPosNum; ++j) {
            for (int k=0; k<plainNumPerPos; ++k) {
                curDstSvv[j].push_back(dstSvv[j][i*plainNumPerPos+k]);
            }
        }
        writeShareVecVectoFile(curDstSvv, fileName);
    }

    return batchSize;
}

uint64_t client_gcn_batch_oblivious_mapper_preprocess(std::vector<uint64_t>& srcPos, std::vector<uint64_t>& dstPos, const std::vector<uint32_t>& plainNumPerPoses,
                                    uint64_t iter, uint64_t preprocessId, const uint32_t dstTid, bool allowMissing) {
    uint64_t srcPosNum = srcPos.size();
    uint64_t dstPosNum = dstPos.size();
	TaskComm& taskComm = TaskComm::getClientInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    uint32_t keySize = cryptoUtil.getKeySize();
    uint32_t plainTextBits = cryptoUtil.getPlainTextBits();
    uint32_t paddingBits = cryptoUtil.getPaddingBits();
    // uint32_t batchSize = keySize / ((plainTextBits + paddingBits) * plainNumPerPos); // Max pos num per cipher entry
    size_t num_layers = plainNumPerPoses.size();
    uint64_t plainNumPerPos = 0;
    for (int i = 0; i < num_layers; ++i) plainNumPerPos += plainNumPerPoses[i];
    for (int i = 0; i < num_layers; ++i) printf("%u ", plainNumPerPoses[i]);
    printf("\n");
    uint32_t batchEpochSize = FHE_SLOT_NUM / plainNumPerPos; // batch of epochs
    uint32_t numEpochIters = num_layers;
    uint32_t batchSize = batchEpochSize * numEpochIters; // batch of GAS iterations
    uint32_t plainNumPerCipherEntry = plainNumPerPos * batchEpochSize;

    // Build mapping from srcPos value to srcPos index
    std::unordered_map<uint64_t, uint64_t> posMap;
    for (int i=0; i<srcPosNum; ++i) {
        // Insert the first index of a value only
        if (posMap.find(srcPos[i]) == posMap.end())
            posMap.insert({srcPos[i], i});
    }

    taskComm.sendObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);
    
    // Initialize s (random ShareVecVec of length dstPosNum)
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    // Recv srcCev
    std::vector<std::string> srcCev;
    taskComm.recvCiphertextVec(srcCev, dstTid);
    // Do mapping & random share splitting
    std::vector<std::string> dstCev;
    dstCev.resize(dstPosNum);

    // #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing client split random share (s) i=%d\n", i); 
        auto search = posMap.find(dstPos[i]);
        uint64_t srcIndex = 0;
        if (search == posMap.end()) {
            if (!allowMissing) {
                printf("Did not find pos index in posMap! %lld\n", dstPos[i]);
                exit(-1);
            }
        } else {
            srcIndex = search->second;
        }
        dstCev[i] = srcCev[srcIndex];
        // printf("party %d client, srcCev.ceVecs[srcIndex].tid %d dstCev.ceVecs[i].tid %d\n", tileIndex, srcCev.ceVecs[srcIndex].tid, dstCev.ceVecs[i].tid);
        cryptoUtil.fhe_mut.lock();
        cryptoUtil.splitRandomShareFromTroyFHECipherEntry(dstCev[i], dstTid, plainNumPerCipherEntry, dstSvv[i]);
        cryptoUtil.fhe_mut.unlock();
    }

    // Send dstCev
    taskComm.sendCiphertextVec(dstCev, dstTid);   

    std::cout<<"client dstSvv[0].size() = "<<dstSvv[0].size()<<std::endl;

    for (int i=0; i<batchEpochSize; ++i) {
        size_t lineOffset = 0;
        for (int j=0; j<numEpochIters; ++j) {
            if (plainNumPerPoses[j] == 0) continue;
            // Store dstSvv (s)
            std::string fileName = "client_dstSvv_";
            fileName = fileName + std::to_string(iter + i*numEpochIters + j) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
            ShareVecVec curDstSvv;
            curDstSvv.resize(dstPosNum); 
            for (int k=0; k<dstPosNum; ++k) {
                curDstSvv[k].clear();
                curDstSvv[k].insert(
                    curDstSvv[k].begin(), 
                    dstSvv[k].begin() + i*plainNumPerPos + lineOffset, 
                    dstSvv[k].begin() + i*plainNumPerPos + lineOffset + plainNumPerPoses[j]
                );
            }
            writeShareVecVectoFile(curDstSvv, fileName);

            lineOffset += plainNumPerPoses[j];
        } 
    }

    return batchSize;
}

uint64_t server_batch_oblivious_mapper_preprocess(uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    uint64_t srcPosNum = 0;
    uint64_t dstPosNum = 0;
    uint64_t plainNumPerPos = 0;
	TaskComm& taskComm = TaskComm::getServerInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    taskComm.recvObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);

    uint32_t keySize = cryptoUtil.getKeySize();
    uint32_t plainTextBits = cryptoUtil.getPlainTextBits();
    uint32_t paddingBits = cryptoUtil.getPaddingBits();
    // uint32_t batchSize = keySize / ((plainTextBits + paddingBits) * plainNumPerPos); // Max pos num per cipher entry
#ifdef USE_FHE 
    uint32_t batchSize = FHE_SLOT_NUM / plainNumPerPos;
#else
    uint32_t batchSize = 1;
#endif
    uint32_t plainNumPerCipherEntry = plainNumPerPos * batchSize;

    // Generate r (random ShareVecVec of length srcPosNum)
    ShareVecVec srcSvv;
    srcSvv.resize(srcPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        srcSvv[i].resize(plainNumPerCipherEntry);
        for (int j=0; j<plainNumPerCipherEntry; ++j) {
            srcSvv[i][j] = cryptoUtil.getRandomShare();
        }
    }

    // Encrypt srcSvv
    CipherEntryVec srcCev;
    srcCev.ceVecs.resize(srcPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server encrypt random share ([r]) i=%d\n", i); 
        // srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry(srcSvv[i], tileIndex);
#ifdef USE_FHE
        srcCev.ceVecs[i] = cryptoUtil.encryptToFHECipherEntry(srcSvv[i], tileIndex);
        // printf("Party %d server srcCev.ceVecs[i].tid %d\n", tileIndex, srcCev.ceVecs[i].tid);
#else
        // srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry(srcSvv[i], tileIndex);
        srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry_Mock(srcSvv[i], tileIndex);
#endif
    }    

    // Send srcCev
    taskComm.sendCipherEntryVec(srcCev, dstTid);

    // Recv dstCev
    CipherEntryVec dstCev;
    taskComm.recvCipherEntryVec(dstCev, dstTid);
    if (dstCev.ceVecs.size() != dstPosNum) {
        printf("Illegal received dstCev size!\n");
        exit(-1);
    }

    // Do decryption
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)  
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server decrypt random share (M*r-s) i=%d\n", i); 
        // cryptoUtil.decryptCipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
#ifdef USE_FHE
        cryptoUtil.decryptFHECipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
#else
        // cryptoUtil.decryptCipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
        cryptoUtil.decryptCipherEntryToInt64Vec_Mock(dstCev.ceVecs[i], dstSvv[i]);
#endif
    }

    for (int i=0; i<batchSize; ++i) {
        // Store dstSvv (M*r-s)
        std::string fileName = "server_dstSvv_";
        fileName = fileName + std::to_string(iter + i) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
        ShareVecVec curDstSvv;
        curDstSvv.resize(dstPosNum);
        for (int j=0; j<dstPosNum; ++j) {
            for (int k=0; k<plainNumPerPos; ++k) {
                curDstSvv[j].push_back(dstSvv[j][i*plainNumPerPos+k]);
            }
        }
        writeShareVecVectoFile(curDstSvv, fileName);

        // Store srcSvv (r)
        fileName = "server_srcSvv_";
        fileName = fileName + std::to_string(iter + i) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
        ShareVecVec curSrcSvv;
        curSrcSvv.resize(srcPosNum);
        for (int j=0; j<srcPosNum; ++j) {
            for (int k=0; k<plainNumPerPos; ++k) {
                curSrcSvv[j].push_back(srcSvv[j][i*plainNumPerPos+k]);
            }
        }
        writeShareVecVectoFile(curSrcSvv, fileName);
    }

    return batchSize;
}

uint64_t server_gcn_batch_oblivious_mapper_preprocess(const std::vector<uint32_t>& plainNumPerPoses, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    uint64_t srcPosNum = 0;
    uint64_t dstPosNum = 0;
    uint64_t plainNumPerPos = 0;
	TaskComm& taskComm = TaskComm::getServerInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();

    taskComm.recvObliviousMapperConfig(srcPosNum, dstPosNum, plainNumPerPos, dstTid);

    uint32_t keySize = cryptoUtil.getKeySize();
    uint32_t plainTextBits = cryptoUtil.getPlainTextBits();
    uint32_t paddingBits = cryptoUtil.getPaddingBits();
    // uint32_t batchSize = keySize / ((plainTextBits + paddingBits) * plainNumPerPos); // Max pos num per cipher entry
    size_t num_layers = plainNumPerPoses.size();
    plainNumPerPos = 0;
    for (int i = 0; i < num_layers; ++i) plainNumPerPos += plainNumPerPoses[i];
    // for (int i = 0; i < num_layers; ++i) printf(">> plainNumPerPoses[i] = %d ", plainNumPerPoses[i]);
    // printf("\n");
    uint32_t batchEpochSize = FHE_SLOT_NUM / plainNumPerPos; // batch of epochs
    uint32_t numEpochIters = num_layers;
    uint32_t batchSize = batchEpochSize * numEpochIters; // batch of GAS iterations
    uint32_t plainNumPerCipherEntry = plainNumPerPos * batchEpochSize;

    // Generate r (random ShareVecVec of length srcPosNum)
    ShareVecVec srcSvv;
    srcSvv.resize(srcPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        srcSvv[i].resize(plainNumPerCipherEntry);
        for (int j=0; j<plainNumPerCipherEntry; ++j) {
            srcSvv[i][j] = cryptoUtil.getRandomShare();
        }
    }

    // Encrypt srcSvv
    std::vector<std::string> srcCev;
    srcCev.resize(srcPosNum);

    // #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server encrypt random share ([r]) i=%d\n", i); 
        // srcCev.ceVecs[i] = cryptoUtil.encryptToCipherEntry(srcSvv[i], tileIndex);
        cryptoUtil.fhe_mut.lock();
        srcCev[i] = cryptoUtil.encryptToTroyFHECipherEntry(srcSvv[i]);
        cryptoUtil.fhe_mut.unlock();
    }    

    // Send srcCev
    taskComm.sendCiphertextVec(srcCev, dstTid);

    // Recv dstCev
    std::vector<std::string> dstCev;
    taskComm.recvCiphertextVec(dstCev, dstTid);
    if (dstCev.size() != dstPosNum) {
        printf("Illegal received dstCev size!\n");
        exit(-1);
    }

    // Do decryption
    ShareVecVec dstSvv;
    dstSvv.resize(dstPosNum);

    // #pragma omp parallel for num_threads(MY_THREAD_NUM)  
    for (int i=0; i<dstPosNum; ++i) {
        if (i%10000 == 0) printf("OM preprocessing server decrypt random share (M*r-s) i=%d\n", i); 
        // cryptoUtil.decryptCipherEntryToInt64Vec(dstCev.ceVecs[i], dstSvv[i]);
        cryptoUtil.fhe_mut.lock();
        cryptoUtil.decryptTroyFHECipherEntryToUint64Vec(dstCev[i], plainNumPerCipherEntry, dstSvv[i]);
        cryptoUtil.fhe_mut.unlock();
    }

    for (int i=0; i<batchEpochSize; ++i) {
        size_t lineOffset = 0;
        for (int j=0; j<numEpochIters; ++j) {
            if (plainNumPerPoses[j] == 0) continue;
            // Store dstSvv (M*r-s)
            std::string fileName = "server_dstSvv_";
            fileName = fileName + std::to_string(iter + i*numEpochIters + j) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
            ShareVecVec curDstSvv;
            curDstSvv.resize(dstPosNum);
            for (int k=0; k<dstPosNum; ++k) {
                curDstSvv[k].clear();
                curDstSvv[k].insert(
                    curDstSvv[k].begin(), 
                    dstSvv[k].begin() + i*plainNumPerPos + lineOffset, 
                    dstSvv[k].begin() + i*plainNumPerPos + lineOffset + plainNumPerPoses[j]
                );
            }
            writeShareVecVectoFile(curDstSvv, fileName);

            // Store srcSvv (r)
            fileName = "server_srcSvv_";
            fileName = fileName + std::to_string(iter + i*numEpochIters + j) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
            ShareVecVec curSrcSvv;
            curSrcSvv.resize(srcPosNum);
            for (int k=0; k<srcPosNum; ++k) {
                curSrcSvv[k].clear();
                curSrcSvv[k].insert(
                    curSrcSvv[k].begin(), 
                    srcSvv[k].begin() + i*plainNumPerPos + lineOffset, 
                    srcSvv[k].begin() + i*plainNumPerPos + lineOffset + plainNumPerPoses[j]
                );
            }

            writeShareVecVectoFile(curSrcSvv, fileName);

            lineOffset += plainNumPerPoses[j];
        }
    }

    return batchSize;
}

void client_oblivious_mapper_online(std::vector<uint64_t>& srcPos, std::vector<uint64_t>& dstPos,
                                    const ShareVecVec& srcSvv, ShareVecVec& dstSvv,
                                    uint64_t plainNumPerPos, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid, bool allowMissing) {
    uint64_t srcPosNum = srcPos.size();
    uint64_t dstPosNum = dstPos.size();
	TaskComm& taskComm = TaskComm::getClientInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    
    // Build mapping from srcPos value to srcPos index
    std::unordered_map<uint64_t, uint64_t> posMap;
    for (int i=0; i<srcPosNum; ++i) {
        // Insert the first index of a value only
        if (posMap.find(srcPos[i]) == posMap.end())
            posMap.insert({srcPos[i], i});
    }

    // Load preprocessed dstSvv (s)
    ShareVecVec preDstSvv;
    std::string fileName = "client_dstSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    loadShareVecVecFromFile(preDstSvv, fileName);
    if (preDstSvv[0].size() < plainNumPerPos) {
        printf("Client preDstSvv plainNumPerPos not enough\n");
        exit(-1);
    }
    for (int i=0; i<dstPosNum; ++i) {
        preDstSvv[i].resize(plainNumPerPos);
    }

    // Recv server srcSvv (<x>_1-r)
    ShareVecVec serverSrcSvv;
    taskComm.recvShareVecVec(serverSrcSvv, dstTid);

    // Merge srcSvv with server srcSvv
    ShareVecVec tmpSvv;
    tmpSvv = srcSvv;
    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        cryptoUtil.addShareVec(tmpSvv[i], srcSvv[i], serverSrcSvv[i]);
    }

    // Do mapping & merge preprocessed dstSvv
    dstSvv.clear();
    dstSvv.resize(dstPosNum);

    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<dstPosNum; ++i) {
        auto search = posMap.find(dstPos[i]);
        uint64_t srcIndex = 0;
        if (search == posMap.end()) {
            if (!allowMissing) {
                printf("Did not find pos index in posMap! %lld\n", dstPos[i]);
                exit(-1);
            }
        } else {
            srcIndex = search->second;
        }
        dstSvv[i].resize(plainNumPerPos);
        cryptoUtil.addShareVec(dstSvv[i], tmpSvv[srcIndex], preDstSvv[i]);
    }
}

void server_oblivious_mapper_online(const ShareVecVec& srcSvv, ShareVecVec& dstSvv,
                                    uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    uint64_t srcPosNum = 0;
    uint64_t dstPosNum = 0;
	TaskComm& taskComm = TaskComm::getServerInstance();
	size_t tileNum = taskComm.getTileNum();
	size_t tileIndex = taskComm.getTileIndex();
    CryptoUtil& cryptoUtil = CryptoUtil::getInstance();
    uint64_t plainNumPerPos = srcSvv[0].size();

    // Load preprocessed srcSvv
    ShareVecVec preSrcSvv;
    std::string fileName = "server_srcSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    loadShareVecVecFromFile(preSrcSvv, fileName);
    srcPosNum = preSrcSvv.size();
    if (preSrcSvv[0].size() < plainNumPerPos) {
        printf("Server preSrcSvv plainNumPerPos not enough preSrcSvv[0].size() = %d, plainNumPerPos = %d\n", preSrcSvv[0].size(), plainNumPerPos);
        exit(-1);
    }
    for (int i=0; i<srcPosNum; ++i) {
        preSrcSvv[i].resize(plainNumPerPos);
    }

    // Load preprocessed dstSvv ad dstSvv
    fileName = "server_dstSvv_";
    fileName = fileName + std::to_string(iter) + "_" + std::to_string(preprocessId) + "_" + std::to_string(tileIndex) + "_" + std::to_string(dstTid);
    loadShareVecVecFromFile(dstSvv, fileName);
    dstPosNum = dstSvv.size();
    if (dstSvv[0].size() < plainNumPerPos) {
        printf("Server dstSvv plainNumPerPos not enough\n");
        exit(-1);
    }
    for (int i=0; i<dstPosNum; ++i) {
        dstSvv[i].resize(plainNumPerPos);
    }

    // srcSvv minus preprocessed secSvv
    ShareVecVec tmpSvv;
    tmpSvv = srcSvv;
    // printf("srcSvv size = %lu\n", srcSvv.size());
    // printf("preSrcSvv size = %lu\n", preSrcSvv.size());
    #pragma omp parallel for num_threads(MY_THREAD_NUM)
    for (int i=0; i<srcPosNum; ++i) {
        cryptoUtil.substractShareVec(tmpSvv[i], srcSvv[i], preSrcSvv[i]);
    }

    // Send server tmpSvv
    taskComm.sendShareVecVec(tmpSvv, dstTid);
}

uint64_t client_gcn_batch_oblivious_shuffle_preprocess(std::vector<uint64_t>& shuffle, const std::vector<uint32_t>& plainNumPerPoses, 
                                    uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    size_t length = shuffle.size();
    std::vector<uint64_t> default_src(length, 0); 
    for (int i = 0; i < length; ++i) default_src[i] = i;

    client_gcn_batch_oblivious_mapper_preprocess(default_src, shuffle, plainNumPerPoses, 
                                    iter, preprocessId, dstTid, false);
    return server_gcn_batch_oblivious_mapper_preprocess(plainNumPerPoses, iter, preprocessId, dstTid);
}

uint64_t server_gcn_batch_oblivious_shuffle_preprocess(std::vector<uint64_t>& shuffle, const std::vector<uint32_t>& plainNumPerPoses, 
                                    uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    size_t length = shuffle.size();
    std::vector<uint64_t> default_src(length, 0); 
    for (int i = 0; i < length; ++i) default_src[i] = i;

    server_gcn_batch_oblivious_mapper_preprocess(plainNumPerPoses, iter, preprocessId, dstTid);
    
    return client_gcn_batch_oblivious_mapper_preprocess(default_src, shuffle, plainNumPerPoses, 
                                    iter, preprocessId, dstTid, false);
}

void client_oblivious_shuffle_online(std::vector<uint64_t>& shuffle,
                                    const ShareVecVec& srcSvv, ShareVecVec& dstSvv,
                                    uint64_t plainNumPerPos, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    size_t length = shuffle.size();
    std::vector<uint64_t> default_src(length, 0); 
    for (int i = 0; i < length; ++i) default_src[i] = i;
    
    ShareVecVec newSvv1;
    ShareVecVec newSvv2;

    client_oblivious_mapper_online(default_src, shuffle,
                                    srcSvv, newSvv1,
                                    plainNumPerPos, iter, preprocessId, dstTid, false);

    server_oblivious_mapper_online(newSvv1, newSvv2,
                                    iter, preprocessId, dstTid);

    dstSvv.swap(newSvv2);
}

void server_oblivious_shuffle_online(std::vector<uint64_t>& shuffle,
                                    const ShareVecVec& srcSvv, ShareVecVec& dstSvv,
                                    uint64_t plainNumPerPos, uint64_t iter, uint64_t preprocessId, const uint32_t dstTid) {
    size_t length = shuffle.size();
    std::vector<uint64_t> default_src(length, 0); 
    for (int i = 0; i < length; ++i) default_src[i] = i;

    ShareVecVec newSvv1;
    ShareVecVec newSvv2;

    server_oblivious_mapper_online(srcSvv, newSvv1,
                                    iter, preprocessId, dstTid);

    client_oblivious_mapper_online(default_src, shuffle,
                                    newSvv1, newSvv2,
                                    plainNumPerPos, iter, preprocessId, dstTid, false);

    dstSvv.swap(newSvv2);
}

void writeShareVecVectoFile(ShareVecVec& svv, std::string fileName) {
    TaskComm& taskComm = TaskComm::getClientInstance();
    std::string dirName = "preprocess/";
    dirName += (taskComm.getSetting() + "/");
    fileName = dirName + fileName;
    std::ofstream ofile;
    ofile.open(fileName);
    if (ofile.is_open()) {
        boost::archive::binary_oarchive oa(ofile);  
        oa << svv;
        ofile.flush();
    } else {
        std::cout<<"Unable to open output file "<<fileName<<std::endl;
    }
}

void loadShareVecVecFromFile(ShareVecVec& svv, std::string fileName) {
    TaskComm& taskComm = TaskComm::getClientInstance();
    std::string dirName = "preprocess/";
    dirName += (taskComm.getSetting() + "/");
    fileName = dirName + fileName;
    std::ifstream ifile;
    ifile.open(fileName);
    if (ifile.is_open()) {
        boost::archive::binary_iarchive ia(ifile);
        svv.clear();
        ia >> svv;
    } else {
        std::cout<<"Unable to open input file "<<fileName<<std::endl;
    }
}