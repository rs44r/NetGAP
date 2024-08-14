import networkx as nx
import torch
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
from torch.nn import Linear,  BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from GraphState import GraphState, GraphEvaluator
from GeneticEvaluator import GeneticEvaluator



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,conv_layers,lin_layers,node_features):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.convs_1 = torch.nn.ModuleList()
        self.bns_1 = torch.nn.ModuleList()
        self.lins_1=torch.nn.ModuleList()

        self.convs_1.append(GCNConv(1, hidden_channels))
        self.bns_1.append(BatchNorm1d(hidden_channels))
        for i in range (conv_layers-1):
          self.convs_1.append(GCNConv(hidden_channels, hidden_channels))
          self.bns_1.append(BatchNorm1d(hidden_channels))



        for i in range (lin_layers-1):
          self.lins_1.append(Linear(hidden_channels,hidden_channels))
        self.lin_out = Linear(hidden_channels,node_features)



    def forward(self, x, edge_index, batch, edge_weights=None,):
        # 1. Obtain node embeddings


        for step in range(len(self.convs_1)):
           x = F.relu(self.bns_1[step](self.convs_1[step](x, edge_index)))

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]


        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        for step in range(len(self.lins_1)):
          x= self.lins_1[step](x)
          x = x.relu()

        x = self.lin_out(x)
        return x



class HybridEvaluator(GraphEvaluator):
    def __init__(self,filename,weights,seed=None,threshold=0.7):
        #self.model = GCN(hidden_channels=64,node_features=1)
        self.model =  GCN(hidden_channels=64,node_features=1,conv_layers=5,lin_layers=3)
        self.model.load_state_dict(torch.load(filename,map_location=torch.device('cpu')))
        self.model.eval()
        self.module_threshold = 22
        self.genetic_evaluator=GeneticEvaluator(weights=weights, pop_size=50, generations=3,filename = "C:/Users\smora\Documents\LiU\Workspace\\NetGAP\data\connections.json",seed=seed)
        self.ref_threshold=threshold
        
    def __str__(self) -> str:
        return str(self.model)
    
    def update_seed(self,seed):
        self.genetic_evaluator.update_seed(seed=seed)
        
    def update_threshold(self,th):
        self.ref_threshold = th

    def trim_graph(self, graph):
        graph
        nodes_to_remove=[]
        for x,y in graph.nodes(data=True):
            if graph.degree(x) >2 and y['idType']=='M':
                edges_to_remove=[(x,z) for z  in graph.neighbors(x)][0:-1]+[(z,x) for z in graph.neighbors(x)][0:-1]
                graph.remove_edges_from(edges_to_remove)
            if graph.degree(x) <=2 and y['idType']=='S':
                nodes_to_remove.append(x)
            elif graph.degree(x) == 4 and y['idType']=='S':
                switches=[n for n in graph.neighbors(x) if graph.nodes[n]["idType"]=='S']
                modules=[n for n in graph.neighbors(x) if graph.nodes[n]["idType"]=='M']
                if len(switches)==1 and len(modules)==1:
                    nodes_to_remove.append(x)
                    graph.add_edge(switches[0],modules[0])
                    graph.add_edge(modules[0],switches[0])
        graph.remove_nodes_from(nodes_to_remove)
    
    def prepare_data(self,graph):
        
        nodes={label:n for n,label in enumerate(graph.nodes)}
        x=torch.tensor([0 if att["idType"] and att["idType"]=='S' else 1 for _,att in graph.nodes(data=True)],dtype=torch.float).reshape([-1, 1])
        edge_index=torch.tensor([[nodes[label] for label,_ in  graph.edges],[nodes[label] for _,label in  graph.edges]],dtype=torch.long)
        
        return Data(edge_index=edge_index, x=x )
    
    def evaluate(self,state):
  
        self.trim_graph(state.graph)
        data = self.prepare_data(state.graph)
        score = self.model(data.x,data.edge_index,data.batch).item()
        if score > self.ref_threshold:
            return self.genetic_evaluator.evaluate(state)
        else:
            return score, {'score':score}
        
    def is_terminal(self, state) -> bool:

        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) >= self.module_threshold
        
        return number_of_modules 