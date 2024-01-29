#include "graph_analyzer.h"

// void test_roc_auc_score() {
//     std::vector<double> y_true_1 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     std::vector<double> y_hat_1 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1}; // {0.1, 0.9, 0.2, 0.8, 0.7};
//     double expected_1 = 0.847368;
//     double actual_1 = roc_auc_score(y_true_1, y_hat_1);
//     if (actual_1 != expected_1) {
//         printf("Error actual_1 = %f, expected_1 = %f \n", actual_1, expected_1);
//         exit(-1);
//     } 
// }

int main(int argc, char* argv[]) {
    int partyNum = 2;
    int partyId = (int)strtol(argv[1], NULL, 10);
    int role;
    if (partyId == 0) role = sci::ALICE;
    else role = sci::BOB;

    // check if enough arguments are provided
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " partyId GNNConfigFile v_path e_path setting n_epochs do_prep is_cluster\n";
        return 1;
    }

    // read the pdf GNNConfigFile, v_path, e_path, n_epochs, do_prep from argv
    std::string GNNConfigFile = argv[2];
    std::string v_path = argv[3];
    std::string e_path = argv[4];
    std::string setting = argv[5];
    size_t n_epochs = (size_t)strtol(argv[6], NULL, 10);
    bool do_prep = ((size_t)strtol(argv[7], NULL, 10) == 1);
    bool is_cluster = ((size_t)strtol(argv[8], NULL, 10) == 1);

    // std::string GNNConfigFile = "/home/zzh/project/test-GCN/Art/CoGNN/tools/config/cora_config.txt";
    // std::string v_path = "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.vertex.preprocessed";
    // std::string e_path = "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.edge.preprocessed";
    // size_t n_epochs = 1;
    // size_t do_prep = true;

    TaskComm& clientTaskComm = TaskComm::getClientInstance();
    clientTaskComm.tileNumIs(partyNum);
    clientTaskComm.tileIndexIs(partyId);
    clientTaskComm.settingIs(setting);
    clientTaskComm.isClusterIs(is_cluster);

    printf("HERE1\n");

    TaskComm& serverTaskComm = TaskComm::getServerInstance();
    serverTaskComm.tileNumIs(partyNum);
    serverTaskComm.tileIndexIs(partyId);
    serverTaskComm.settingIs(setting);
    serverTaskComm.isClusterIs(is_cluster);

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

    // test_roc_auc_score();

    // std::string GNNConfigFile = "/home/zzh/project/test-GCN/Art/CoGNN/tools/config/cora_small_config.txt";

    GNNParam& gnnParam = GNNParam::getGNNParam();
    gnnParam.readConfig(GNNConfigFile);

    GraphParam param = GraphParam {
        num_iters: gnnParam.num_layers * 2 * n_epochs,
        do_preprocess: do_prep
    };

    // std::string v_path = "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.vertex.small";
    // std::string e_path = "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.edge.small";

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;
    std::vector<std::vector<double>> E;
    read_data(v_path, e_path, X, Y, E);

    // std::string file_path = "./../test-data/small_graph.csv";
    DataFrame df; // load_dataframe_from_csv(file_path, false, false);

    GraphAnalyzer ana(
        E,
        X,
        Y,
        param,
        partyId,
        role
    );

    ana.run();

    sci::closeSCIChannel(role, 1 - partyId);

    clientTaskComm.closeChannels();
    serverTaskComm.closeChannels();
    
    return 0;
}