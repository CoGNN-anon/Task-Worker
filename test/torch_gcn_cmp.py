import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

def print_grad(grad, name):
    print(f"Gradient of {name}:")
    print(grad)

def print_mask(data):
    print('train_mask', torch.count_nonzero(data.train_mask))
    print('val_mask', torch.count_nonzero(data.val_mask))
    print('test_mask', torch.count_nonzero(data.test_mask))

def set_mask(data, train_ratio, val_ratio, test_ratio, device):
    num_samples = data.x.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    data.train_mask = torch.tensor([True] * num_train + [False] * num_val + [False] * num_test).to(device)
    data.val_mask = torch.tensor([False] * num_train + [True] * num_val + [False] * num_test).to(device)
    data.test_mask = torch.tensor([False] * num_train + [False] * num_val + [True] * num_test).to(device)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels, bias = False)
        torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        # self.conv1.lin.weight.data = torch.full_like(self.conv1.lin.weight, 0.5)
        # self.conv1.lin.weight.register_hook(lambda grad: print_grad(grad, "conv1.weight"))

        self.conv2 = GCNConv(hidden_channels, num_classes, bias = False)
        torch.nn.init.xavier_uniform_(self.conv2.lin.weight)
        # self.conv2.lin.weight.data = torch.full_like(self.conv2.lin.weight, 0.5)
        # self.conv2.lin.weight.register_hook(lambda grad: print_grad(grad, "conv2.weight"))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # print(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x.register_hook(lambda grad: print_grad(grad, "before softmax"))
        # return F.softmax(x, dim=1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def load_cora():
#     # Load Cora dataset
#     dataset = Planetoid(root='data', name='Cora')
#     model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes).to(device)
#     data = dataset[0].to(device)
#     print_mask(data)
#     set_mask(data, 0.2, 0.2, 0.6, device)
#     print_mask(data)
#     return data, model

def load_tiny():
    # Read the edge file and convert it to a tensor of shape [2, num_edges]
    edge_file = "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.edge.small" # Change this to your edge file path
    edge_index = torch.tensor([list(map(int, line.split())) for line in open(edge_file)], dtype=torch.long).t()

    # Read the node file and convert it to a tensor of shape [num_nodes, num_features]
    node_file = "/home/zzh/project/test-GCN/FedGCNData/data/tiny/cora.vertex.small" # Change this to your node file path
    x = torch.tensor([list(map(float, line.split()[1:-1])) for line in open(node_file)], dtype=torch.float)

    # Read the node class file and convert it to a tensor of shape [num_nodes]
    y = torch.tensor([int(line.split()[-1]) for line in open(node_file)], dtype=torch.long)
    # Create a data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.Tensor(), val_mask=torch.Tensor(), test_mask=torch.Tensor()).to(device)

    print(data)
    print(type(data))

    set_mask(data, 1.0, 0.0, 0.0, device)    
    model = GCN(num_features=2, hidden_channels=3, num_classes=3).to(device)
    return data, model

def load_cora():
    # Read the edge file and convert it to a tensor of shape [2, num_edges]
    edge_file = "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.edge.preprocessed" # Change this to your edge file path
    edge_index = torch.tensor([list(map(int, line.split())) for line in open(edge_file)], dtype=torch.long).t()

    # Read the node file and convert it to a tensor of shape [num_nodes, num_features]
    node_file = "/home/zzh/project/test-GCN/FedGCNData/data/cora/cora.vertex.preprocessed" # Change this to your node file path
    x = torch.tensor([list(map(float, line.split()[1:-1])) for line in open(node_file)], dtype=torch.float)

    # Read the node class file and convert it to a tensor of shape [num_nodes]
    y = torch.tensor([int(line.split()[-1]) for line in open(node_file)], dtype=torch.long)
    # Create a data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.Tensor(), val_mask=torch.Tensor(), test_mask=torch.Tensor()).to(device)

    # print(data)
    # print(type(data))

    set_mask(data, 0.2, 0.2, 0.6, device)    
    model = GCN(num_features=x.shape[1], hidden_channels=16, num_classes=7).to(device)
    return data, model

def load_citeseer():
    # Read the edge file and convert it to a tensor of shape [2, num_edges]
    edge_file = "/home/zzh/project/test-GCN/FedGCNData/data/citeseer/citeseer.edge.preprocessed" # Change this to your edge file path
    edge_index = torch.tensor([list(map(int, line.split())) for line in open(edge_file)], dtype=torch.long).t()

    # Read the node file and convert it to a tensor of shape [num_nodes, num_features]
    node_file = "/home/zzh/project/test-GCN/FedGCNData/data/citeseer/citeseer.vertex.preprocessed" # Change this to your node file path
    x = torch.tensor([list(map(float, line.split()[1:-1])) for line in open(node_file)], dtype=torch.float)

    # Read the node class file and convert it to a tensor of shape [num_nodes]
    y = torch.tensor([int(line.split()[-1]) for line in open(node_file)], dtype=torch.long)
    # Create a data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.Tensor(), val_mask=torch.Tensor(), test_mask=torch.Tensor()).to(device)

    # print(data)
    # print(type(data))

    set_mask(data, 0.2, 0.2, 0.6, device)    
    model = GCN(num_features=x.shape[1], hidden_channels=16, num_classes=7).to(device)
    return data, model

def load_pubmed():
    # Read the edge file and convert it to a tensor of shape [2, num_edges]
    edge_file = "/home/zzh/project/test-GCN/FedGCNData/data/pubmed/pubmed.edge.preprocessed" # Change this to your edge file path
    edge_index = torch.tensor([list(map(int, line.split())) for line in open(edge_file)], dtype=torch.long).t()

    # Read the node file and convert it to a tensor of shape [num_nodes, num_features]
    node_file = "/home/zzh/project/test-GCN/FedGCNData/data/pubmed/pubmed.vertex.preprocessed" # Change this to your node file path
    x = torch.tensor([list(map(float, line.split()[1:-1])) for line in open(node_file)], dtype=torch.float)

    # Read the node class file and convert it to a tensor of shape [num_nodes]
    y = torch.tensor([int(line.split()[-1]) for line in open(node_file)], dtype=torch.long)
    # Create a data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.Tensor(), val_mask=torch.Tensor(), test_mask=torch.Tensor()).to(device)

    # print(data)
    # print(type(data))

    set_mask(data, 0.05, 0.15, 0.8, device)    
    model = GCN(num_features=x.shape[1], hidden_channels=16, num_classes=7).to(device)
    return data, model

# data, model = load_tiny()
# data, model = load_cora()
# data, model = load_citeseer()
data, model = load_pubmed()
print(type(data))

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=5)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # print(out)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

for epoch in range(180):
    loss = train()
    acc = test()
    # acc = 0.0
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')