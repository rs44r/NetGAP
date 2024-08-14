class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,node_features):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin0 = Linear(hidden_channels,node_features)

    def forward(self, x, edge_index,  batch, edge_weight=None):
        # 1. Obtain node embeddings
        if edge_weight is not None:
          x = self.conv1(x, edge_index, edge_weight)
          x = x.relu()
          x = self.conv2(x, edge_index, edge_weight)
          x = x.relu()
          x = self.conv3(x, edge_index, edge_weight)
          x = x.relu()
          x = self.conv4(x, edge_index, edge_weight)
          x = x.relu()
        else:
          x = self.conv1(x, edge_index)
          x = x.relu()
          x = self.conv2(x, edge_index)
          x = x.relu()
          x = self.conv3(x, edge_index)
          x = x.relu()
          x = self.conv4(x, edge_index)
          x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final regression layer
        x = self.lin0(x)

        return x