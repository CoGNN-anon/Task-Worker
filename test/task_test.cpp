#include "TaskqHandler.h"
#include "TaskUtil.h"
#include "SCIHarness.h"
#include "task.h"

#include <vector>

void test_ssGCNVectorScale(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_ssGCNVectorScale\n");

    size_t taskNum = 2;
    std::vector<std::vector<double>> embeddingVecs_plain = {
        {10, 12, 4, 5, 2},
        {1, 2, 3, 4, 5}
    };
    std::vector<std::vector<double>> expected = embeddingVecs_plain;
    std::vector<double> scaler0_plain = {
        2,
        3
    };
    std::vector<double> scaler1_plain = {
        0.3,
        0.1
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

    std::vector<Task> taskVec;
    for (int i = 0; i < taskNum; ++i) {
        GCNVectorScale* gvs = new GCNVectorScale();
        gvs->writeShareToOperand(embeddingVecs[i], scaler0[i], scaler1[i]);
        struct Task task = {GCN_VECTOR_SCALE, 0, false, (void*)gvs, 0, 1-partyId, partyId};
        taskVec.push_back(task);
    } 

    sci::ssGCNVectorScale(taskVec, 1-partyId, role);

    for (int i = 0; i < taskNum; ++i) {
        embeddingVecs[i] = ((GCNVectorScale*)taskVec[i].buf)->embeddingVec;
        taskVec[i].delete_task_content_buf();
    } 

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
    } else {
        serverTaskComm.sendShareVecVec(embeddingVecs, 1 - partyId);
    }
}

void test_ssGCNVectorAddition(TaskComm& clientTaskComm, TaskComm& serverTaskComm, int partyId, int role) {
    printf(">>>> test_ssGCNVectorAddition\n");

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

    std::vector<Task> taskVec;
    for (int i = 0; i < taskNum; ++i) {
        GCNVectorAddition* gva = new GCNVectorAddition();
        gva->operands[0] = operands0[i];
        gva->operands[1] = operands1[i];     
        gva->operandMask = true;
        struct Task task = {GCN_VECTOR_ADDITION, 0, false, (void*)gva, static_cast<uint64_t>(-1), 1-partyId, partyId, 0, true, false};
        taskVec.push_back(task);
    } 

    sci::ssGCNVectorAddition(taskVec, 1-partyId, role);

    for (int i = 0; i < taskNum; ++i) {
        operands0[i] = ((GCNVectorAddition*)taskVec[i].buf)->operands[0];
        taskVec[i].delete_task_content_buf();
    } 

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

    test_ssGCNVectorScale(clientTaskComm, serverTaskComm, partyId, role);
    test_ssGCNVectorAddition(clientTaskComm, serverTaskComm, partyId, role);

    sci::closeSCIChannel(role, 1 - partyId);

    clientTaskComm.closeChannels();
    serverTaskComm.closeChannels();
    
    return 0;
}