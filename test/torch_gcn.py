from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# def print_grad(grad):
#     print(grad)

def print_grad(grad, name):
    print(f"Gradient of {name}:")
    print(grad)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        # self.lin.weight.register_hook(print_grad)
        torch.nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # x.register_hook(lambda grad: print_grad(grad, "before softmax"))

        return F.softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
                'node_size': 30,
                'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))

def train(data, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        # print(out.detach())
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # for param in model.parameters():
        #     print(param.grad)

        optimizer.step()

        train_acc = test(data)
        test_acc = test(data, train=False)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                format(epoch, loss, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()

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

if __name__ == "__main__":
    dataset = Planetoid(root='data', name='Cora')
    
    # plot_dataset(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset).to(device)
    data = dataset[0].to(device)
    # print(data.x.tolist())
    print_mask(data)
    set_mask(data, 0.6, 0.2, 0.2, device)
    print_mask(data)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    train(data, plot=True)