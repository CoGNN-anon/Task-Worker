#include "task.h"
#include "TaskUtil.h"

CryptoUtil& TaskPayload::cryptoUtil = CryptoUtil::getInstance();

CipherEntry SwapCipherEntry::splitShareFromEncryptedOperand(int operandId0) {
    uint64_t share = cryptoUtil.splitRandomShareFromCipherEntry(operands[operandId0].enc);
    // operands[operandId0].enc.share.resize(1);
    operands[operandId0].enc.share[0] = share;
    operands[operandId0].enc.isShare = true;
    return operands[operandId0].enc;
}

CipherEntry SwapCipherEntry::encryptShare(int operandId0, const uint32_t tid) {
    CipherEntry ce = cryptoUtil.encryptToCipherEntry(operands[operandId0].enc.share[0], tid);
    return ce;
}

void SwapCipherEntry::mergeEncryptedShare(CipherEntry& ce, int operandId0) {
    cryptoUtil.mergeShareIntoCipherEntry(ce, operands[operandId0].enc.share[0], ce.tid);
    operands[operandId0].enc = ce;
    operands[operandId0].enc.isShare = false;
    isOperandEncrypted[operandId0] = true;
    return;
}

void SwapCipherEntry::writeShareToOperand(uint64_t share, int operandId0, int operandId1) {
    // operands[operandId0].enc.share.resize(1);
    operands[operandId0].enc.share[0] = share;
    operands[operandId0].enc.isShare = true;
    return;
}

ShareVec SwapCipherEntry::getOperandShare(int operandId0) {
    ShareVec sv;
    sv.push_back(operands[operandId0].enc.share[0]);
    return sv;
}

int SwapCipherEntry::getEncTid(int operandId0) {
    return operands[operandId0].enc.tid;
}

void SwapCipherEntry::writeEncryptedOperand(CipherEntry& ce, int operandId0) {
    operands[operandId0].enc = ce;
    operands[operandId0].enc.isShare = false;
    isOperandEncrypted[operandId0] = true;
}

CipherEntry* SwapCipherEntry::getCipherEntryPtr(int operandId0) {
    return &operands[operandId0].enc;
}

void SwapCipherEntry::unifyOperand() {
    operands[1] = operands[0];
    isOperandEncrypted[1] = isOperandEncrypted[0];
}

void SwapCipherEntry::copyOperand(TaskPayload& srcTP, int srcOperandId0, int dstOperandId0) {
    operands[dstOperandId0] = static_cast<SwapCipherEntry&>(srcTP).operands[srcOperandId0];
}

void SwapCipherEntry::setCipherEntryPlainNum(int operandId0, int num) {
    operands[operandId0].enc.plainNum = num;
}

ShareTensor transpose(const ShareTensor& st) {
    ShareTensor result(st[0].size(), std::vector<uint64_t>(st.size()));
    for (int i = 0; i < st[0].size(); i++) 
        for (int j = 0; j < st.size(); j++) {
            result[i][j] = st[j][i];
        }
    return result;
}

ShareTensor toShareTensor(const ShareVec& sv) {
    ShareTensor curST;
    curST.push_back(sv);
    return curST;
}

ShareVec toShareVec(const ShareTensor& st) {
    if (st.size() != 1) {
        printf("Illegal conversion from ShareTensor to ShareVector!\n");
        exit(-1);
    }
    ShareVec sv;
    sv = st[0];
    return sv;
}

ShareVec toShareVec(int hotIndex, int vecSize) {
    ShareVec sv;
    sv.resize(vecSize, 0);
    sv[hotIndex] = (1 << SCALER_BIT_LENGTH);
    return sv;
}