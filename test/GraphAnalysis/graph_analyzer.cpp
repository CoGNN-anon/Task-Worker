#include "graph_analyzer.h"

#include <cassert>
#include <set>
#include <chrono>
#include <math.h>

// Read X, Y from a vertex file and A from an edge file function
void read_data(std::string vertex_file, std::string edge_file, std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& Y, std::vector<std::vector<double>>& E) {
  GNNParam& gnnParam = GNNParam::getGNNParam();
  // Open the vertex file
  std::ifstream vfile(vertex_file);
  if (!vfile.is_open()) {
    std::cerr << "Error: cannot open the vertex file" << std::endl;
    return;
  }

  // Initialize the vertex feature matrix X and the label matrix Y as zero matrices
  X = std::vector<std::vector<double>>(gnnParam.num_samples, std::vector<double>(gnnParam.input_dim, 0.0));
  Y = std::vector<std::vector<double>>(gnnParam.num_samples, std::vector<double>(gnnParam.num_labels, 0.0));

  // Read the vertex file line by line
  std::string line;
  while (std::getline(vfile, line)) {
    // Split the line by whitespace
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, ' ')) {
      tokens.push_back(token);
    }
    // Check if the line has the correct format
    if (tokens.size() != gnnParam.input_dim + 2) {
      std::cerr << "Error: invalid vertex file format" << std::endl;
      return;
    }
    // Get the sample index, the features, and the label from the line
    int index = std::stoi(tokens[0]);
    std::vector<double> features(gnnParam.input_dim, 0.0);
    std::vector<double> label(gnnParam.num_labels, 0.0);
    for (int i = 0; i < gnnParam.input_dim; i++) {
      features[i] = std::stod(tokens[i + 1]);
    }
    int class_index = std::stoi(tokens[gnnParam.input_dim + 1]);
    label[class_index] = 1.0;
    // Assign the features and the label to the corresponding row of X and Y
    X[index] = features;
    Y[index] = label;
  }

  // Close the vertex file
  vfile.close();

  // Open the edge file
  std::ifstream efile(edge_file);
  if (!efile.is_open()) {
    std::cerr << "Error: cannot open the edge file" << std::endl;
    return;
  }

  E.clear();

  // Read the edge file line by line
  while (std::getline(efile, line)) {
    // Split the line by whitespace
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, ' ')) {
      tokens.push_back(token);
    }
    // Check if the line has the correct format
    if (tokens.size() != 2) {
      std::cerr << "Error: invalid edge file format" << std::endl;
      return;
    }
    // Get the indices of the two vertices from the line
    int i = std::stoi(tokens[0]);
    int j = std::stoi(tokens[1]);
    // Assign 1 to the corresponding element of A
    E.push_back({(double)i, (double)j});
  }

  // Close the edge file
  efile.close();

  // Return nothing
  return;
}

std::vector<double> get_vertex_degree(const std::vector<std::vector<double>>& _e, size_t num_vertices) {
    std::vector<double> result(num_vertices, 0);
    for (auto& x : _e) {
        result[(uint64_t)x[0]] += 1;
    } 
    return result;
}

void get_randomized_permutation(std::vector<size_t>& permutation, size_t size) {
	permutation.resize(size);

	for (int i = 0; i < size; ++i)
		permutation[i] = i;

	osuCrypto::PRNG prng(_mm_set_epi32(4253233465, 334565, 0, 235)); // we need to modify this seed

	for (int i = size - 1; i > 0; i--)
	{
		int loc = prng.get<uint64_t>() % (i + 1); //  pick random location in the array
		std::swap(permutation[i], permutation[loc]);
	}
}

void get_randomized_permutation_with_inverse(std::vector<size_t>& permutation, std::vector<size_t>& inv_permutation, size_t size) {
    get_randomized_permutation(permutation, size);
    inv_permutation.resize(size);
	for (int i=0; i < size; ++i) {
		inv_permutation[permutation[i]] = i;
	}    
}

void get_inverse_permutation(const std::vector<size_t>& permutation, std::vector<size_t>& inv_permutation) {
    size_t perm_size = permutation.size();
    inv_permutation.resize(perm_size);
	for (int i=0; i < perm_size; ++i) {
		inv_permutation[permutation[i]] = i;
	}    
}

void open_permute(std::vector<std::vector<uint64_t>>& v_share, const std::vector<size_t>& perm) {
    assert(v_share.size() == perm.size());
    size_t perm_size = perm.size();
    std::vector<std::vector<uint64_t>> result(perm_size);
    for (int i = 0; i < perm_size; ++i) {
        result[i] = v_share[perm[i]];
    }
    v_share.swap(result);
}

void open_permute(std::vector<uint64_t>& v_share, const std::vector<size_t>& perm) {
    assert(v_share.size() == perm.size());
    size_t perm_size = perm.size();
    std::vector<uint64_t> result(perm_size);
    for (int i = 0; i < perm_size; ++i) {
        result[i] = v_share[perm[i]];
    }
    v_share.swap(result);
}


GraphAnalyzer::GraphAnalyzer(const std::vector<std::vector<double>>& _e, const std::vector<std::vector<double>>& _x, const std::vector<std::vector<double>>& _y, GraphParam _gp, int _party_id, int _party) {
    assert(_e.size() > 0);
    assert(_e[0].size() == 2);
    e_list = _e;
    x_list = _x;
    y_list = _y;
    deg_list = get_vertex_degree(e_list, x_list.size());
    param = _gp;
    party_id = _party_id;
    party = _party;
}

void GraphAnalyzer::construct_table_from_raw_graph() {
    GNNParam& gnnParam = GNNParam::getGNNParam();
    // print_vector_of_vector(e_list, 10);
    // std::set<uint64_t> vertex_index_set;
    size_t num_edges = e_list.size();
    size_t num_vertices = x_list.size();
    table.resize(num_edges);
    if (e_list[0].size() == 2) { // Only vertex indices are provided in the raw graph
        for (int i = 0; i < num_edges; ++i) {
            uint64_t srcDegReci = CryptoUtil::encodeDoubleAsFixedPoint(pow((double)deg_list[(uint64_t)e_list[i][0]] + 1, -0.5));
            uint64_t dstDegReci = CryptoUtil::encodeDoubleAsFixedPoint(pow((double)deg_list[(uint64_t)e_list[i][1]] + 1, -0.5));
            table[i] = {CryptoUtil::encodeDoubleAsFixedPoint(e_list[i][0]), CryptoUtil::encodeDoubleAsFixedPoint(e_list[i][1]), CryptoUtil::encodeDoubleAsFixedPoint(1), srcDegReci, dstDegReci};
            std::vector<uint64_t> dummy_edge_row(gnnParam.input_dim - 2, 0);
            table[i].insert(table[i].end(), dummy_edge_row.begin(), dummy_edge_row.end());
            // vertex_index_set.insert((uint64_t)e_list[i][0]);
            // vertex_index_set.insert((uint64_t)e_list[i][1]);
        }
        for (int i = 0; i < num_vertices; ++i) {
            table.push_back({CryptoUtil::encodeDoubleAsFixedPoint(i), CryptoUtil::encodeDoubleAsFixedPoint(i), (uint64_t)0});
            size_t cur_index = table.size() - 1;
            table[cur_index].insert(table[cur_index].end(), x_list[i].begin(), x_list[i].end());
        }
    }
    // print_vector_of_vector(table, table.size());
    table_size = table.size();
    printf("whole table size = %ld\n",table_size);
    printf("number of edges = %ld\n", num_edges);
    printf("number of vertices = %ld\n", table_size - num_edges);
    // table = transpose(table);
    Ws.resize(gnnParam.num_layers);
    Ws[0].resize(gnnParam.input_dim, std::vector<uint64_t>(gnnParam.hidden_dim, 0));
    Ws[1].resize(gnnParam.hidden_dim, std::vector<uint64_t>(gnnParam.num_labels, 0));
}

void GraphAnalyzer::generate_shuffle_and_preprocess() {
    GNNParam& gnnParam = GNNParam::getGNNParam();
    get_randomized_permutation_with_inverse(shuffle1, inv_shuffle1, table_size);
    get_randomized_permutation_with_inverse(shuffle2, inv_shuffle2, table_size);

    // // Get channels
    // if (party == sci::ALICE) {
    //     TaskComm& taskComm = TaskComm::getClientInstance();
    //     chls.push_back(*taskComm.getChannel(1 - party_id));
    // } else {
    //     TaskComm& taskComm = TaskComm::getServerInstance();
    //     chls.push_back(*taskComm.getChannel(1 - party_id));   
    // }

    // shuffle1_ps.is_preprocessed = false;
    // shuffle1_pr.is_preprocessed = false;
    // inv_shuffle1_ps.is_preprocessed = false;
    // inv_shuffle1_pr.is_preprocessed = false;
    // shuffle2_ps.is_preprocessed = false;
    // shuffle2_pr.is_preprocessed = false;
    // inv_shuffle2_ps.is_preprocessed = false;
    // inv_shuffle2_pr.is_preprocessed = false;

    auto t_shuffle_preprocess = std::chrono::high_resolution_clock::now();
    // Preprocess
    // std::vector<uint64_t> tmp(table_size);
    common_src.resize(table_size);
    for (int i = 0; i < table_size; ++i) common_src[i] = i;

    if (!param.do_preprocess) return;
    
    std::vector<uint32_t> shuffle1_plainNumPerPoses = {gnnParam.input_dim + 3, gnnParam.hidden_dim + 3, gnnParam.num_labels + 3, gnnParam.hidden_dim + 3};
    std::vector<uint32_t> shuffle2_plainNumPerPoses = {gnnParam.input_dim, gnnParam.hidden_dim, gnnParam.num_labels, gnnParam.hidden_dim};
    uint64_t iter = 0;

    if (party == sci::ALICE) {
        client_gcn_batch_oblivious_shuffle_preprocess(shuffle1, shuffle1_plainNumPerPoses, 0, 0, 1 - party_id);
        server_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle1, shuffle1_plainNumPerPoses, 0, 2, 1 - party_id);
    } else {
        server_gcn_batch_oblivious_shuffle_preprocess(shuffle1, shuffle1_plainNumPerPoses, 0, 0, 1 - party_id);
        client_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle1, shuffle1_plainNumPerPoses, 0, 2, 1 - party_id);
    }

    while (iter < param.num_iters) {
        if (party == sci::ALICE) {
            // // ALICE's shuffle1 first, then BOB's shuffle1
            // client_gcn_batch_oblivious_shuffle_preprocess(shuffle1, plainNumPerPoses, iter, 0, 1 - party_id);
            // // ALICE's shuffle2 first, then BOB's shuffle2
            size_t batch_size = client_gcn_batch_oblivious_shuffle_preprocess(shuffle2, shuffle2_plainNumPerPoses, iter, 1, 1 - party_id);
            // // BOB's inv_shuffle1 first, then ALICE's inv_shuffle1
            // server_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle1, plainNumPerPoses, iter, 2, 1 - party_id);
            // // BOB's inv_shuffle2 first, then ALICE's inv_shuffle2
            server_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle2, shuffle2_plainNumPerPoses, iter, 3, 1 - party_id);
            iter += batch_size;
        } else {
            // server_gcn_batch_oblivious_shuffle_preprocess(shuffle1, plainNumPerPoses, iter, 0, 1 - party_id);
            size_t batch_size = server_gcn_batch_oblivious_shuffle_preprocess(shuffle2, shuffle2_plainNumPerPoses, iter, 1, 1 - party_id);
            // client_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle1, plainNumPerPoses, iter, 2, 1 - party_id);
            client_gcn_batch_oblivious_shuffle_preprocess(inv_shuffle2, shuffle2_plainNumPerPoses, iter, 3, 1 - party_id);
            iter += batch_size;
        }
    }
    print_duration(t_shuffle_preprocess, "duration shuffle preprocess");
}

void concat_src_with_is_edge(const std::vector<std::vector<uint64_t>>& table, std::vector<uint64_t>& concat) {
    assert(table.size() >= 3);
    assert(table[0].size() > 0);
    size_t table_size = table[0].size();
    concat.resize(table_size);
    for (int i = 0; i < table_size; ++i) {
        concat[i] = (table[0][i] << 1) + table[2][i];
    }
}

void concat_dst_with_is_edge(const std::vector<std::vector<uint64_t>>& table, std::vector<uint64_t>& concat) {
    assert(table.size() >= 3);
    assert(table[0].size() > 0);
    size_t table_size = table[0].size();
    concat.resize(table_size);
    for (int i = 0; i < table_size; ++i) {
        concat[i] = (table[1][i] << 1) + table[2][i];
    }
}

void GraphAnalyzer::vectorized_scatter(std::vector<std::vector<uint64_t>>& propagated_vertex_cols, std::vector<std::vector<uint64_t>>& data_cols) {
    size_t length = data_cols.size();
    std::vector<std::vector<uint64_t>> scattered;
    std::vector<uint64_t> srcDeg(length, 0);
    std::vector<uint64_t> dstDeg(length, 0);
    sci::twoPartyGCNVectorScale(
        data_cols, 
        srcDeg, 
        dstDeg, 
        scattered, 
        1 - party_id, 
        party
    );
    // scattered = data_cols;
    // sci::twoParty(data_cols, propagated_vertex_col, scattered, party_id, party);
    // sci::twoPartyMux2(data_col, scattered, is_edge_before_Scatter, party_id, party);
    sci::twoPartyMux2(
        data_cols, 
        scattered, 
        is_edge_before_Scatter, 
        data_cols, 
        1 - party_id, 
        party, 
        false
    );
}

void GraphAnalyzer::vectorized_gather(std::vector<uint64_t>& dst_vertex_col, std::vector<std::vector<uint64_t>>& data_cols) {
    size_t length = data_cols.size();
    std::vector<uint64_t> srcDeg(length, 0);
    std::vector<uint64_t> dstDeg(length, 0);
    sci::twoPartyGCNVectorScale(
        data_cols, 
        srcDeg, 
        dstDeg, 
        data_cols, 
        1 - party_id, 
        party
    );
    std::vector<std::vector<uint64_t>> agg_result = prefix_network_aggregate(dst_vertex_col, data_cols, AggregationOp::ADD_AGG, 1 - party_id, party);
    // sci::twoPartySelectedAssign(data_col, agg_result, is_vertex_before_Gather, party_id, party);
    sci::twoPartyMux2(
        data_cols, 
        agg_result, 
        is_vertex_before_Gather, 
        data_cols, 
        1 - party_id, 
        party, 
        false
    );
}

void GraphAnalyzer::vectorized_apply(std::vector<std::vector<uint64_t>>& data_cols, uint64_t iter) {
    GNNParam& gnnParam = GNNParam::getGNNParam();
    size_t layer_index = iter % (gnnParam.num_layers * 2);
    size_t length = data_cols.size();
    std::vector<uint64_t> normalizer;
    std::vector<std::vector<uint64_t>> applied;
    if (layer_index == 0) {
        std::vector<std::vector<uint64_t>> z;
        std::vector<std::vector<uint64_t>> new_h;
        sci::twoPartyGCNForwardNN(data_cols, Ws[0], normalizer, z, new_h, 1 - party_id, party);
        applied.swap(new_h);
    } else if (layer_index == 1) {
        std::vector<std::vector<uint64_t>> p;
        std::vector<std::vector<uint64_t>> z;
        std::vector<std::vector<uint64_t>> p_minus_y;
        std::vector<std::vector<uint64_t>> label(length, std::vector<uint64_t>(gnnParam.num_labels, 0));
        sci::twoPartyGCNForwardNNPrediction(data_cols, Ws[1], label, normalizer, z, p, p_minus_y, 1 - party_id, party);
        applied.swap(p_minus_y);
    } else if (layer_index == 2) {
        std::vector<std::vector<uint64_t>> ah_t(gnnParam.hidden_dim, std::vector<uint64_t>(length, 0));
        std::vector<std::vector<uint64_t>> weightT = transpose(Ws[1]);
        std::vector<std::vector<uint64_t>> d;
        std::vector<std::vector<uint64_t>> g;
        sci::twoPartyGCNBackwardNNInit(data_cols, ah_t, weightT, normalizer, d, g, 1 - party_id, party);
        double gradientScaler = (double) 1;
        double learningRate = (double) 0.01;
        sci::twoPartyGCNMatrixScale(d, CryptoUtil::encodeDoubleAsFixedPoint(gradientScaler), d, 1 - party_id, party);
        sci::twoPartyGCNApplyGradient(Ws[1], d, CryptoUtil::encodeDoubleAsFixedPoint(learningRate), Ws[1], 1 - party_id, party);    
        applied.swap(g);
    } else if (layer_index == 3) {
        std::vector<std::vector<uint64_t>> ah_t(gnnParam.input_dim, std::vector<uint64_t>(length, 0));
        std::vector<std::vector<uint64_t>> weightT = transpose(Ws[0]);
        std::vector<std::vector<uint64_t>> d;
        std::vector<std::vector<uint64_t>> g;
        std::vector<std::vector<uint64_t>> z(length, std::vector<uint64_t>(gnnParam.hidden_dim, 0));;
        sci::twoPartyGCNBackwardNN(data_cols, ah_t, z, weightT, normalizer, d, g, true, 1 - party_id, party);
        double gradientScaler = (double) 1;
        double learningRate = (double) 0.01;
        sci::twoPartyGCNMatrixScale(d, CryptoUtil::encodeDoubleAsFixedPoint(gradientScaler), d, 1 - party_id, party);
        sci::twoPartyGCNApplyGradient(Ws[0], d, CryptoUtil::encodeDoubleAsFixedPoint(learningRate), Ws[0], 1 - party_id, party);  
        applied.resize(length, std::vector<uint64_t>(gnnParam.num_labels, 0));
    } else {
        printf("Unexpected layer during Apply!\n");
        exit(-1);
    }

    data_cols.swap(applied);

    // sci::twoPartyMux2(
    //     data_cols, 
    //     applied, 
    //     is_vertex_before_Gather, 
    //     data_cols, 
    //     1 - party_id, 
    //     party, 
    //     false
    // );
}

void GraphAnalyzer::run() {
    construct_table_from_raw_graph();
    generate_shuffle_and_preprocess();
    GNNParam& gnnParam = GNNParam::getGNNParam();

    std::vector<uint32_t> shuffle1_plainNumPerPoses = {gnnParam.input_dim + 3, gnnParam.hidden_dim + 3, gnnParam.num_labels + 3, gnnParam.hidden_dim + 3};
    std::vector<uint32_t> shuffle2_plainNumPerPoses = {gnnParam.input_dim, gnnParam.hidden_dim, gnnParam.num_labels, gnnParam.hidden_dim};

    // Shuffle 1
    if (party == sci::ALICE) {
        // ALICE's shuffle1 first, then BOB's shuffle1
        client_oblivious_shuffle_online(shuffle1,
                                    table, table,
                                    shuffle1_plainNumPerPoses[0], 0, 0, 1 - party_id);
    } else {
        server_oblivious_shuffle_online(shuffle1,
                                    table, table,
                                    shuffle1_plainNumPerPoses[0], 0, 0, 1 - party_id);
    }

    std::vector<std::vector<uint64_t>> table_T = transpose(table);

    auto t_open_sort = std::chrono::high_resolution_clock::now();

    // Get two open sorts
    std::vector<uint64_t> concat_src;
    concat_src_with_is_edge(table_T, concat_src);
    open_sort_by_src = common_src;
    // ss_open_sort(
    //     concat_src,
    //     open_sort_by_src,
    //     party_id,
    //     party
    // );
    get_inverse_permutation(open_sort_by_src, inv_open_sort_by_src);

    std::vector<uint64_t> concat_dst;
    concat_dst_with_is_edge(table_T, concat_dst);
    assert(concat_dst.size() == table_size);
    std::vector<std::vector<uint64_t>> concat_dst_2d(table_size, std::vector<uint64_t>(shuffle2_plainNumPerPoses[0], 0));
    for (int i = 0; i < table_size; ++i) concat_dst_2d[i][0] = concat_dst[i];
    // Shuffle 2
    if (party == sci::ALICE) {
        // ALICE's shuffle2 first, then BOB's shuffle2
        client_oblivious_shuffle_online(shuffle2,
                                    concat_dst_2d, concat_dst_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
    } else {
        server_oblivious_shuffle_online(shuffle2,
                                    concat_dst_2d, concat_dst_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
    }
    for (int i = 0; i < table_size; ++i) concat_dst[i] = concat_dst_2d[i][0];
    open_sort_by_dst = common_src;
    // ss_open_sort(
    //     concat_dst,
    //     open_sort_by_dst,
    //     party_id,
    //     party
    // );
    get_inverse_permutation(open_sort_by_dst, inv_open_sort_by_dst);

    print_duration(t_open_sort, "t_open_sort");

    // Record src before Scatter & dst before Gather
    src_before_Scatter = table_T[0];
    open_permute(src_before_Scatter, open_sort_by_src);
    dst_before_Gather = table_T[1];
    std::vector<std::vector<uint64_t>> dst_before_Gather_2d(table_size, std::vector<uint64_t>(shuffle2_plainNumPerPoses[0], 0));
    for (int i = 0; i < table_size; ++i) dst_before_Gather_2d[i][0] = dst_before_Gather[i];
    // Shuffle 2
    if (party == sci::ALICE) {
        // ALICE's shuffle2 first, then BOB's shuffle2
        client_oblivious_shuffle_online(shuffle2,
                                    dst_before_Gather_2d, dst_before_Gather_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
    } else {
        server_oblivious_shuffle_online(shuffle2,
                                    dst_before_Gather_2d, dst_before_Gather_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
    }
    for (int i = 0; i < table_size; ++i) dst_before_Gather[i] = dst_before_Gather_2d[i][0] ;
    open_permute(dst_before_Gather, open_sort_by_dst);

    // Record is_edge flag before Scatter
    std::vector<uint64_t> tmp_flag = table_T[2];
    open_permute(tmp_flag, open_sort_by_src);
    is_edge_before_Scatter.resize(table_size);
    for (int i = 0; i < table_size; ++i) is_edge_before_Scatter[i] = (uint8_t)(tmp_flag[i] & 0xFF);

    // Record is_vertex flag before Gather
    tmp_flag = table_T[2];
    std::vector<std::vector<uint64_t>> tmp_flag_2d(table_size, std::vector<uint64_t>(shuffle2_plainNumPerPoses[0], 0));
    for (int i = 0; i < table_size; ++i) tmp_flag_2d[i][0] = tmp_flag[i];
    // Shuffle 2
    if (party == sci::ALICE) {
        // ALICE's shuffle2 first, then BOB's shuffle2
        client_oblivious_shuffle_online(shuffle2,
                                    tmp_flag_2d, tmp_flag_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
        for (int i = 0; i < table_size; ++i) tmp_flag[i] = 1 - tmp_flag_2d[i][0]; 
    } else {
        server_oblivious_shuffle_online(shuffle2,
                                    tmp_flag_2d, tmp_flag_2d,
                                    shuffle2_plainNumPerPoses[0], 0, 1, 1 - party_id);
        for (int i = 0; i < table_size; ++i) tmp_flag[i] = - tmp_flag_2d[i][0];
    } 
    open_permute(tmp_flag, open_sort_by_dst);
    is_vertex_before_Gather.resize(table_size);
    for (int i = 0; i < table_size; ++i) is_vertex_before_Gather[i] = (uint8_t)(tmp_flag[i] & 0xFF);

    // GAS iterations
    std::vector<std::vector<uint64_t>> data_cols(table_T.begin() + 3, table_T.end());
    data_cols = transpose(data_cols);
    std::vector<std::vector<uint64_t>> data_cols_backup = data_cols;
    for (int i = 0; i < param.num_iters; ++i) {
        size_t layer_index = i % (gnnParam.num_layers * 2);
        if (i > 0 && layer_index == 0) data_cols = data_cols_backup;
        auto t_iter = std::chrono::high_resolution_clock::now();

        if (layer_index != 2) {
            printf("H0\n");
            // Scatter
            open_permute(data_cols, open_sort_by_src);
            std::vector<std::vector<uint64_t>> propagated_vertex_data = prefix_network_propagate(src_before_Scatter, data_cols, 1 - party_id, party);
            vectorized_scatter(propagated_vertex_data, data_cols);

            // Gather
            open_permute(data_cols, inv_open_sort_by_src);
            // Shuffle 2
            if (party == sci::ALICE) {
                // ALICE's shuffle2 first, then BOB's shuffle2
                client_oblivious_shuffle_online(shuffle2,
                                            data_cols, data_cols,
                                            shuffle2_plainNumPerPoses[layer_index], i, 1, 1 - party_id);
            } else {
                server_oblivious_shuffle_online(shuffle2,
                                            data_cols, data_cols,
                                            shuffle2_plainNumPerPoses[layer_index], i, 1, 1 - party_id);
            }
            open_permute(data_cols, open_sort_by_dst);
            vectorized_gather(dst_before_Gather, data_cols);
        }

        // printf("Before Apply %lu\n", data_cols[0].size());

        vectorized_apply(data_cols, i);

        if (layer_index == 1) {
            print_duration(t_iter, "iteration");
            continue;
        }

        // printf("H1\n");
        // printf("After Apply %lu\n", data_cols[0].size());

        open_permute(data_cols, inv_open_sort_by_dst);

        size_t pos_index_mod = layer_index+1;
        if (layer_index == 3) pos_index_mod = 2;
        size_t iter_shuffle2 = i+1;
        if (layer_index == 3) iter_shuffle2 = 2;
        // Inv Shuffle 2
        if (party == sci::ALICE) {
            // BOB's inv_shuffle2 first, then ALICE's inv_shuffle2
            server_oblivious_shuffle_online(inv_shuffle2,
                                        data_cols, data_cols,
                                        shuffle2_plainNumPerPoses[pos_index_mod], iter_shuffle2, 3, 1 - party_id);
        } else {
            client_oblivious_shuffle_online(inv_shuffle2,
                                        data_cols, data_cols,
                                        shuffle2_plainNumPerPoses[pos_index_mod], iter_shuffle2, 3, 1 - party_id);
        }

        print_duration(t_iter, "iteration");
    }
    // table[3] = data_cols;
    data_cols = transpose(data_cols);
    table_T.resize(3);
    table_T.insert(table_T.end(), data_cols.begin(), data_cols.end());
    table = transpose(table_T);

    // Inv Shuffle 1
    if (party == sci::ALICE) {
        // BOB's inv_shuffle1 first, then ALICE's inv_shuffle1
        // oblivious_shuffle_table_sender(table, inv_shuffle1_pr, inv_shuffle1_ps, common_src, inv_shuffle1, chls);
        server_oblivious_shuffle_online(inv_shuffle1,
                                    table, table,
                                    shuffle1_plainNumPerPoses[2], 2, 2, 1 - party_id);
    } else {
        // oblivious_shuffle_table_receiver(table, inv_shuffle1_pr, inv_shuffle1_ps, common_src, inv_shuffle1, chls);
        client_oblivious_shuffle_online(inv_shuffle1,
                                    table, table,
                                    shuffle1_plainNumPerPoses[2], 2, 2, 1 - party_id);
    }
}