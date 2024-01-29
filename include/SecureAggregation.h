#ifndef SECURE_AGGREGATION_H_
#define SECURE_AGGREGATION_H_

#include <array>

#include "task.h"
#include "SCIHarness.h"

enum AggregationOp {
    NONE_AGG,
    ADD_AGG,
    MIN_AGG,
    MAX_AGG
};

std::vector<std::vector<uint64_t>> prefix_network_aggregate(
    const std::vector<uint64_t>& group_id,
    const std::vector<std::vector<uint64_t>>& value,
    AggregationOp agg_op,
    int coTid,
    int party,
    bool is_group_id_one_side = false
);

std::vector<std::vector<uint64_t>> prefix_network_propagate(
    const std::vector<uint64_t>& group_id,
    const std::vector<std::vector<uint64_t>>& value,
    int coTid,
    int party,
    bool is_group_id_one_side = false
);

#endif