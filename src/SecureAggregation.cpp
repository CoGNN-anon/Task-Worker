#include "SecureAggregation.h"

void get_merge_sequence(
    const std::vector<size_t>& index_vec, 
    std::vector<std::vector<size_t>>& sequence
) {
    size_t length = index_vec.size();
    assert(length > 0);

    if (length == 1) {
        return;
    }

    if (length % 2 == 0) {
        sequence.push_back(index_vec);
    } else {
        sequence.push_back(std::vector<size_t>(index_vec.begin(), index_vec.end() - 1));
    }

    size_t cur_length = (length + 1) / 2;
    std::vector<size_t> cur_index_vec(cur_length);
    for (size_t i = 0; i < cur_length; i++) {
        cur_index_vec[i] = index_vec[i * 2];
    }

    get_merge_sequence(cur_index_vec, sequence);

    if (length <= 2) {
        return;
    }

    if (length % 2 == 0) {
        sequence.push_back(std::vector<size_t>(index_vec.begin() + 1, index_vec.end() - 1));
    } else {
        sequence.push_back(std::vector<size_t>(index_vec.begin() + 1, index_vec.end()));
    }
}

std::vector<uint8_t> get_pair_indicator(
    const std::vector<uint64_t>& group_id_lhs,
    const std::vector<uint64_t>& group_id_rhs,
    int coTid,
    int party,
    bool is_group_id_one_side
) {
    size_t length = group_id_lhs.size();
    assert(length > 0);
    assert(group_id_rhs.size() == length);

    std::vector<uint8_t> eq_result;
    std::vector<uint64_t> array_zero(length, 0);
    if (!is_group_id_one_side) {
        // printf("Here\n");
        if (party == sci::ALICE) sci::twoPartyEq(group_id_lhs, group_id_rhs, eq_result, coTid, party);
        else sci::twoPartyEq(array_zero, array_zero, eq_result, coTid, party);
    } else {
        eq_result.resize(length);
        for (int i = 0; i < length; ++i) {
            eq_result[i] = group_id_lhs[i] == group_id_rhs[i] ? 1 : 0;
        }
    }

    return eq_result;
}

std::vector<std::vector<uint64_t>> conditional_merge(
    const std::vector<uint8_t>& indicator,
    const std::vector<std::vector<uint64_t>>& lhs,
    const std::vector<std::vector<uint64_t>>& rhs,
    AggregationOp agg_op,
    int coTid,
    int party,
    bool is_group_id_one_side
) {
    size_t length = indicator.size();
    assert(length > 0);
    assert(length == lhs.size() && length == rhs.size());

    std::vector<std::vector<uint64_t>> agg_result(length, std::vector<uint64_t>(lhs[0].size(), 0));
    switch (agg_op) {
        case AggregationOp::NONE_AGG:
            throw std::runtime_error("Unsupported Aggregation Op");
        case AggregationOp::ADD_AGG:
            for (int i = 0; i < length; ++i) {
                size_t lineLength = lhs[i].size();
                for (int j = 0; j < lineLength; ++j) {
                    agg_result[i][j] = lhs[i][j] + rhs[i][j];
                }
            }
            break;
        default:
            throw std::runtime_error("Unsupported Aggregation Op");
            break;
    }

    std::vector<std::vector<uint64_t>> cond_result;
    std::vector<uint8_t> zero_indicator(length, 0);
    if (party == sci::ALICE) sci::twoPartyMux2(lhs, agg_result, indicator, cond_result, coTid, party, is_group_id_one_side);
    else sci::twoPartyMux2(lhs, agg_result, zero_indicator, cond_result, coTid, party, is_group_id_one_side);  

    return cond_result;
}

std::vector<std::vector<uint64_t>> prefix_network_aggregate(
    const std::vector<uint64_t>& group_id,
    const std::vector<std::vector<uint64_t>>& value,
    AggregationOp agg_op,
    int coTid,
    int party,
    bool is_group_id_one_side
) {
    size_t length = group_id.size();
    assert(length == value.size());
    assert(length > 0);

    std::vector<std::vector<uint64_t>> ret_value = value;

    if (length == 1) {
        return ret_value;
    }

    std::vector<size_t> index_vec(length);
    std::iota(index_vec.begin(), index_vec.end(), 0); // fill with 0, 1, ..., length - 1
    std::vector<std::vector<size_t>> sequence;
    get_merge_sequence(index_vec, sequence);
    
    std::vector<uint8_t> indicator(length);

    size_t layer_num = sequence.size();

    for (size_t i = 0; i < layer_num; i++) {
        const vector<size_t>& cur_sequence = sequence[i];
        size_t cur_length = cur_sequence.size();
        size_t pair_num = cur_length / 2;
        std::vector<uint64_t> group_id_lhs(pair_num, 0);
        std::vector<uint64_t> group_id_rhs(pair_num, 0);
        std::vector<std::vector<uint64_t>> lhs(pair_num, {0, 0});
        std::vector<std::vector<uint64_t>> rhs(pair_num, {0, 0});

        for (size_t j = 0; j < pair_num; j++) {
            group_id_lhs[j] = group_id[cur_sequence[j * 2]];
            group_id_rhs[j] = group_id[cur_sequence[j * 2 + 1]];
            lhs[j] = ret_value[cur_sequence[j * 2]];
            rhs[j] = ret_value[cur_sequence[j * 2 + 1]];
        }

        std::vector<uint8_t> cur_indicator = get_pair_indicator(group_id_lhs, group_id_rhs, coTid, party, is_group_id_one_side);

        if (i == 0) {
            for (size_t j = 0; j < pair_num; j++) {
                indicator[j * 2] = cur_indicator[j];
            }
        } else if (i == layer_num - 1) {
            for (size_t j = 0; j < pair_num; j++) {
                indicator[j * 2 + 1] = cur_indicator[j];
            }
        }

        std::vector<std::vector<uint64_t>> cur_merge_result = conditional_merge(cur_indicator, lhs, rhs, agg_op, coTid, party, is_group_id_one_side);

        for (size_t j = 0; j < pair_num; j++) {
            ret_value[cur_sequence[j * 2]] = cur_merge_result[j];
        }
    }

    // Set non-start-of-group to zero
    std::vector<uint8_t> filter_indicator = std::vector<uint8_t>(indicator.begin(), indicator.end() - 1);
    std::vector<uint8_t> zero_filter_indicator = std::vector<uint8_t>(filter_indicator.size(), 0);
    std::vector<std::vector<uint64_t>> ret_value_tail(ret_value.begin() + 1, ret_value.end());
    std::vector<std::vector<uint64_t>> zero_vec(length - 1, std::vector<uint64_t>(ret_value[0].size(), 0));
    std::vector<std::vector<uint64_t>> filter_result;
    if (party == sci::ALICE) sci::twoPartyMux2(ret_value_tail, zero_vec, filter_indicator, filter_result, coTid, party, is_group_id_one_side);
    else sci::twoPartyMux2(ret_value_tail, zero_vec, zero_filter_indicator, filter_result, coTid, party, is_group_id_one_side);
    for (size_t i = 0; i < length - 1; i++) {
        ret_value[i + 1] = filter_result[i];
    }

    return ret_value;    
}

std::vector<std::vector<uint64_t>> prefix_network_propagate(
    const std::vector<uint64_t>& group_id,
    const std::vector<std::vector<uint64_t>>& value,
    int coTid,
    int party,
    bool is_group_id_one_side
) {
    size_t length = group_id.size();
    assert(length == value.size());
    assert(length > 0);

    std::vector<std::vector<uint64_t>> ret_value = value;
    size_t row_size = value[0].size();

    if (length == 1) {
        return ret_value;
    }

    // print_decoded_vector(group_id, 10);
    // print_decoded_vector(value, 10);

    std::vector<size_t> index_vec(length);
    std::iota(index_vec.begin(), index_vec.end(), 0); // fill with 0, 1, ..., length - 1
    // print_vector(index_vec, 10);
    std::vector<std::vector<size_t>> sequence;
    get_merge_sequence(index_vec, sequence);

    std::vector<uint8_t> indicator(length);

    size_t layer_num = sequence.size();

    for (int i = layer_num - 1; i >= 0; i--) {
        const vector<size_t>& cur_sequence = sequence[i];
        // printf("%ld\n", i);
        // print_vector(cur_sequence, cur_sequence.size());
        size_t cur_length = cur_sequence.size();
        size_t pair_num = cur_length / 2;
        std::vector<uint64_t> group_id_lhs(pair_num, 0);
        std::vector<uint64_t> group_id_rhs(pair_num, 0);
        std::vector<std::vector<uint64_t>> lhs(pair_num, std::vector<uint64_t>(row_size, 0));
        std::vector<std::vector<uint64_t>> rhs(pair_num, std::vector<uint64_t>(row_size, 0));

        for (size_t j = 0; j < pair_num; j++) {
            group_id_lhs[j] = group_id[cur_sequence[j * 2]];
            group_id_rhs[j] = group_id[cur_sequence[j * 2 + 1]];
            lhs[j] = ret_value[cur_sequence[j * 2]];
            rhs[j] = ret_value[cur_sequence[j * 2 + 1]];
        }

        std::vector<uint8_t> cur_indicator = get_pair_indicator(group_id_lhs, group_id_rhs, coTid, party, is_group_id_one_side);

        std::vector<std::vector<uint64_t>> array_zero(pair_num, std::vector<uint64_t>(row_size, 0));
        std::vector<uint8_t> zero_indicator(pair_num, 0);
        if (party == sci::ALICE) sci::twoPartyMux2(lhs, rhs, cur_indicator, rhs, coTid, party, is_group_id_one_side);
        else sci::twoPartyMux2(lhs, rhs, zero_indicator, rhs, coTid, party, is_group_id_one_side);
        // print_decoded_vector(cur_merge_result, cur_merge_result.size());

        for (size_t j = 0; j < pair_num; j++) {
            ret_value[cur_sequence[j * 2 + 1]] = rhs[j];
        }
    }

    // print_decoded_vector(ret_value, 10);

    return ret_value;
}