#include "task.h"
#include "SCIHarness.h"

#include <vector>

void bitonic_merge(
    size_t lo,
    size_t n,
    bool dir,
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>& seq,
    size_t& depth
);

void bitonic_sort(
    size_t lo,
    size_t n,
    bool dir,
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>& seq,
    size_t& depth
);

void plaintext_bitonic_sort(
    std::vector<int64_t>& inputs
);

void ss_bitonic_sort(
    std::vector<uint64_t>& inputs_share,
    int partyId,
    int role
);

void ss_bitonic_sort(
    std::vector<std::vector<uint64_t>>& inputs_share,
    int partyId,
    int role
);

void plaintext_quick_sort(
    vector<int64_t>& v, 
    int low, 
    int high, 
    vector<size_t>& perm
);

void ss_open_sort(
    vector<uint64_t>& v, 
    vector<size_t>& perm,
    int partyId,
    int role    
);