#include "Sort.h"

size_t greatest_power_of_two_lessthan(size_t n) {
    size_t k = 1;

    while (k < n) {
        k = k << 1;
    }

    return (k >> 1);
}

void bitonic_merge(
    size_t lo,
    size_t n,
    bool dir,
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>& seq,
    size_t& depth
) {
    if (n > 1) {
        size_t m = greatest_power_of_two_lessthan(n);

        if (seq.size() < (depth + 1)) {
            seq.resize(depth + 1, std::make_pair(std::vector<size_t>(), std::vector<size_t>()));
        }

        for (size_t i = lo; i < (lo + n - m); i++) {
            if (dir) {
                seq[depth].first.push_back(i);
                seq[depth].second.push_back(i + m);
            } else {
                seq[depth].first.push_back(i + m);
                seq[depth].second.push_back(i);
            }
        }

        depth = depth + 1;

        size_t lower_depth = depth;
        bitonic_merge(lo, m, dir, seq, lower_depth);

        size_t upper_depth = depth;
        bitonic_merge(lo + m, n - m, dir, seq, upper_depth);

        depth = std::max(lower_depth, upper_depth);
    }
}

void bitonic_sort(
    size_t lo,
    size_t n,
    bool dir,
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>& seq,
    size_t& depth
) {
    if (n > 1) {
        size_t m = n / 2;

        size_t lower_depth = depth;
        bitonic_sort(lo, m, !dir, seq, lower_depth);

        size_t upper_depth = depth;
        bitonic_sort(lo + m, n - m, dir, seq, upper_depth);

        depth = std::max(lower_depth, upper_depth) + 1;

        bitonic_merge(lo, n, dir, seq, depth);
    }
}

void plaintext_cmp_swap(std::vector<int64_t>& inputs, const std::vector<size_t>& lhs_indices, const std::vector<size_t>& rhs_indices) {
    int length = lhs_indices.size();
    for (int i = 0; i < length; ++i) {
        if (inputs[rhs_indices[i]] < inputs[lhs_indices[i]]) {
            int64_t tmp = inputs[lhs_indices[i]];
            inputs[lhs_indices[i]] = inputs[rhs_indices[i]];
            inputs[rhs_indices[i]] = tmp;
        }
    }
}

void ss_cmp_swap(std::vector<uint64_t>& inputs_share, const std::vector<size_t>& lhs_indices, const std::vector<size_t>& rhs_indices, int partyId, int role) {
    int length = lhs_indices.size();
    std::vector<uint64_t> lhs(length);
    std::vector<uint64_t> rhs(length);
    for (int i = 0; i < length; ++i) {
        lhs[i] = inputs_share[lhs_indices[i]];
        rhs[i] = inputs_share[rhs_indices[i]];
    }

    sci::twoPartyCmpSwap(lhs, rhs, partyId, role);

    for (int i = 0; i < length; ++i) {
        inputs_share[lhs_indices[i]] = lhs[i];
        inputs_share[rhs_indices[i]] = rhs[i];
    }    
}

void ss_cmp_swap(std::vector<std::vector<uint64_t>>& inputs_share, const std::vector<size_t>& lhs_indices, const std::vector<size_t>& rhs_indices, int partyId, int role) {
    int length = lhs_indices.size();
    std::vector<std::vector<uint64_t>> lhs(length);
    std::vector<std::vector<uint64_t>> rhs(length);
    for (int i = 0; i < length; ++i) {
        lhs[i] = inputs_share[lhs_indices[i]];
        rhs[i] = inputs_share[rhs_indices[i]];
    }

    sci::twoPartyCmpSwap(lhs, rhs, partyId, role);

    for (int i = 0; i < length; ++i) {
        inputs_share[lhs_indices[i]] = lhs[i];
        inputs_share[rhs_indices[i]] = rhs[i];
    }    
}

void plaintext_bitonic_sort(
    std::vector<int64_t>& inputs
) {
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> sequence;
    size_t depth = 0;
    bitonic_sort(
        0,
        inputs.size(),
        true,
        sequence,
        depth
    );

    depth = sequence.size();
    for (int i = 0; i < depth; ++i) {
        if (sequence[i].first.size() == 0) {
            continue;
        }

        plaintext_cmp_swap(inputs, sequence[i].first, sequence[i].second);
    }
}

void ss_bitonic_sort(
    std::vector<uint64_t>& inputs_share,
    int partyId,
    int role
) {
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> sequence;
    size_t depth = 0;
    bitonic_sort(
        0,
        inputs_share.size(),
        true,
        sequence,
        depth
    );

    depth = sequence.size();
    for (int i = 0; i < depth; ++i) {
        if (sequence[i].first.size() == 0) {
            continue;
        }

        ss_cmp_swap(inputs_share, sequence[i].first, sequence[i].second, partyId, role);
    }
}

void ss_bitonic_sort(
    std::vector<std::vector<uint64_t>>& inputs_share,
    int partyId,
    int role
) {
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> sequence;
    size_t depth = 0;
    bitonic_sort(
        0,
        inputs_share.size(),
        true,
        sequence,
        depth
    );

    depth = sequence.size();
    for (int i = 0; i < depth; ++i) {
        if (sequence[i].first.size() == 0) {
            continue;
        }

        ss_cmp_swap(inputs_share, sequence[i].first, sequence[i].second, partyId, role);
    }
}

// A helper function to swap two elements
void swap(int64_t& a, int64_t& b) {
    int64_t temp = a;
    a = b;
    b = temp;
}

void swap(size_t& a, size_t& b) {
    size_t temp = a;
    a = b;
    b = temp;
}

// A function to partition the vector around a pivot element
size_t plaintext_partition(vector<int64_t>& v, size_t low, size_t high, vector<size_t>& perm) {
    // Choose the last element as the pivot
    int64_t pivot = v[high];
    // Initialize the index of smaller element
    size_t i = low - 1;
    // Loop through the elements from low to high - 1
    for (size_t j = low; j < high; j++) {
        // If the current element is smaller than or equal to the pivot
        if (v[j] <= pivot) {
            // Increment the index of smaller element
            i++;
            // Swap the current element with the element at i
            swap(v[i], v[j]);
            // Swap the corresponding elements in the permutation vector
            swap(perm[i], perm[j]);
        }
    }
    // Swap the pivot element with the element at i + 1
    swap(v[i + 1], v[high]);
    // Swap the corresponding elements in the permutation vector
    swap(perm[i + 1], perm[high]);
    // Return the index of the pivot element
    return i + 1;
}

// A function to perform quick sort on a vector
void plaintext_quick_sort(
    vector<int64_t>& v, 
    int low, 
    int high, 
    vector<size_t>& perm
) {
    // Base case: if the vector has one or zero elements, it is already sorted
    if (low >= high) return;
    // Partition the vector around a pivot element and get its index
    size_t p = plaintext_partition(v, low, high, perm);
    // Recursively sort the left and right subvectors
    plaintext_quick_sort(v, low, p - 1, perm);
    plaintext_quick_sort(v, p + 1, high, perm);
}


size_t ss_partition(vector<uint64_t>& v_share, size_t low, size_t high, vector<size_t>& perm, int partyId, int role) {
    uint64_t pivot = v_share[high];
    size_t i = low - 1;

    size_t length = high - low;
    std::vector<uint64_t> lhs(length);
    std::vector<uint64_t> rhs(length);
    for (int k = 0; k < length; ++k) {
        lhs[k] = pivot;
        rhs[k] = v_share[low + k];
    }

    std::vector<bool> gt_result;
    sci::twoPartyCmpOpen(lhs, rhs, gt_result, partyId, role);

    // printf("Here party %d : \n", partyId);
    // for (int i = 0; i < length; ++i) {
    //     printf("%d ", gt_result[i + low]? 1 : 0);
    // }
    // printf("\n");

    for (size_t j = low; j < high; j++) {
        if (gt_result[j - low]) {
            i++;
            swap(v_share[i], v_share[j]);
            swap(perm[i], perm[j]);
        }
    }

    swap(v_share[i + 1], v_share[high]);
    swap(perm[i + 1], perm[high]);
    return i + 1;
}

void ss_quick_sort(
    vector<uint64_t>& v, 
    int low, 
    int high, 
    vector<size_t>& perm,
    int partyId,
    int role
) {
    if (low >= high) return;
    size_t p = ss_partition(v, low, high, perm, partyId, role);
    ss_quick_sort(v, low, p - 1, perm, partyId, role);
    ss_quick_sort(v, p + 1, high, perm, partyId, role);
}

void ss_open_sort(
    vector<uint64_t>& v, 
    vector<size_t>& perm,
    int partyId,
    int role    
) {
    ss_quick_sort(v, 0, v.size() - 1, perm, partyId, role);
}
