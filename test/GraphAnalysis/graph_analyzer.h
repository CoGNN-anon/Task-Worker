#pragma once

#include "graph_common.h"

typedef std::vector<std::vector<double>> DataFrame;

struct GraphParam {
    int num_iters;
    bool do_preprocess;
};

typedef struct GraphParam GraphParam;

void read_data(std::string vertex_file, std::string edge_file, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& Y, std::vector<std::vector<double>>& E);

class GraphAnalyzer {

public:
    std::vector<std::vector<double>> e_list; // Edge list
    std::vector<std::vector<double>> x_list; // feature list
    std::vector<std::vector<double>> y_list; // feature list
    std::vector<double> deg_list;
    std::vector<std::vector<uint64_t>> table; // Containing both vertices and edges
    std::vector<std::vector<std::vector<uint64_t>>> Ws;
    int party_id;
    int party;
    std::vector<osuCrypto::Channel> chls;
    GraphParam param;

    GraphAnalyzer(const std::vector<std::vector<double>>& _e, const std::vector<std::vector<double>>& _x, const std::vector<std::vector<double>>& _y, GraphParam _gp, int _party_id, int _party);
    void construct_table_from_raw_graph();
    void generate_shuffle_and_preprocess();
    void vectorized_scatter(std::vector<std::vector<uint64_t>>& propagated_vertex_cols, std::vector<std::vector<uint64_t>>& data_col);
    void vectorized_gather(std::vector<uint64_t>& dst_vertex_cols, std::vector<std::vector<uint64_t>>& data_col);
    void vectorized_apply(std::vector<std::vector<uint64_t>>& data_cols, uint64_t iter);
    void run();

    // PermSender shuffle1_ps;
    // PermReceiver shuffle1_pr;
    // PermSender inv_shuffle1_ps;
    // PermReceiver inv_shuffle1_pr;
    // PermSender shuffle2_ps;
    // PermReceiver shuffle2_pr;
    // PermSender inv_shuffle2_ps;
    // PermReceiver inv_shuffle2_pr;

    std::vector<size_t> common_src;
    std::vector<size_t> shuffle1;
    std::vector<size_t> inv_shuffle1;
    std::vector<size_t> shuffle2;
    std::vector<size_t> inv_shuffle2;

    std::vector<size_t> open_sort_by_src;
    std::vector<size_t> inv_open_sort_by_src;
    std::vector<size_t> open_sort_by_dst;
    std::vector<size_t> inv_open_sort_by_dst;

    std::vector<uint64_t> src_before_Scatter;
    std::vector<uint8_t> is_edge_before_Scatter;
    std::vector<uint64_t> dst_before_Gather;
    std::vector<uint8_t> is_vertex_before_Gather;

    size_t table_size;
};