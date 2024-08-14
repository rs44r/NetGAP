from __future__ import annotations


from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from networkx.algorithms import graph_hashing as nxh
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.cm as cm
import numpy as np
import json


from MCSTree import *
from GraphBuilder import *
import networkx as nx
import copy as cp
from typing import TYPE_CHECKING, Tuple, Dict, List

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem



class GraphEvaluator:
    def evaluate(self, state: GraphState) -> Tuple[float, dict]:
        return (0,{})

    def is_terminal(self, state: GraphState) -> bool:
        return True


class GraphState(State):

    #TODO: implement a reward stats print function
    #TODO: evaluator and builder should not be a part of graph state

    graph: nx.DiGraph
    reward: float 
    depth: int
    
    def __deepcopy__(self,memo={}):
        
                
        copy = GraphState(graph_evaluator=self.graph_evaluator, graph_builder=self.graph_builder)
        
        
        copy.depth=self.depth
        copy.reward_stats = {l:v for l,v in self.reward_stats.items()} 
        copy.reward = self.reward
        
        copy.graph = nx.DiGraph()
        copy.graph.add_nodes_from(list(self.graph.nodes(data=True)))
        copy.graph.add_edges_from(list(self.graph.edges(data=True)))
        
        return copy
        
    

    def __init__(self, graph_evaluator: GraphEvaluator, graph_builder: GraphBuilder, graph: Optional[nx.DiGraph] = None, depth: int = 0 ):

        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph
            
        self.depth=depth
        self.reward_stats = {}
        self.reward: float = 0
        #self.hash_number = self.get_hash_number()
        self.graph_evaluator = graph_evaluator
        self.graph_builder = graph_builder
        self.current_player=1


    def get_reward(self) -> float:
        """Calculates reward from evaluator_class"""
        self.reward, self.reward_stats = self.graph_evaluator.evaluate(self)
        return self.reward

    def is_terminal(self) -> bool:
        """ Evaluates if terminal from evaluator_class"""
        no_actions = len(self.get_possible_actions())==0
        evaluate_terminal = self.graph_evaluator.is_terminal(self)
        # if evalaute_terminal | no_actions:
        #     print(no_actions, evalaute_terminal, evalaute_terminal | no_actions)
        return evaluate_terminal | no_actions

    def get_possible_actions(self) -> SupportsLenAndGetItem[Action]:

        return self.graph_builder.match(self)
        
    def take_action(self, a: Action) -> State:
        self.depth+=1
        ret =  self.graph_builder.produce(self,a)
        return ret


    def __hash__(self):
        return int( nxh.weisfeiler_lehman_graph_hash(self.graph, node_attr="idType"),16)
        #return nxh.weisfeiler_lehman_graph_hash(self.graph, node_attr="idType")

    def draw(self, type_map=None, node_attribute_label=None, node_attribute=None, edge_attribute=None, size=(3, 6),
             arc_rad=0,icon_map= None):

        fig=plt.figure(figsize=size)
        plt.axis('off')
        plt.title("Platform Graph")

        top = cm.get_cmap('summer', 256)
        bottom = cm.get_cmap('autumn', 256)
        new_colors = np.vstack((top(np.linspace(0, 0.8, 6)),
                                bottom(np.linspace(1, 0, 4))))
        new_cmp = ListedColormap(new_colors, name='RedGreen')

        sm = plt.cm.ScalarMappable(cmap=new_cmp, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array(np.array([]))

        pos : Optional[Dict[int, Tuple[float,float]]]= graphviz_layout(self.graph, prog="neato")
        pos_offset = {}
        if pos:
            for k, v in pos.items():
                pos_offset[k] = (v[0] * 0.75 + 12, v[1] * 0.75 - 12)

        if not type_map and not node_attribute:
            nx.draw_networkx_nodes(self.graph, pos)

        elif type_map and not node_attribute:
            for key, value in type_map.items():
                nx.draw_networkx_nodes(self.graph, pos, nodelist=[x for x, y in self.graph.nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] == key],
                                       node_color=value[0], node_shape=value[1])

            nx.draw_networkx_nodes(self.graph, pos, nodelist=[x for x, y in self.graph.nodes(data=True) 
                                                              if y and 'idType' in y and y['idType'] not in type_map.keys()],
                                   node_color="red", node_shape='X')

        elif not type_map and node_attribute:
            node_att = nx.get_node_attributes(self.graph, node_attribute)
            nx.draw_networkx_nodes(self.graph, pos, cmap=new_cmp, vmin=0, vmax=1, node_color=node_att)

        elif type_map and node_attribute:
            print("should be here")
            for key, value in type_map.items():
                if value[0] == 'attribute':
                    nodes, node_values = zip(*nx.get_node_attributes(self.graph, node_attribute).items())
                    value[0] = node_values
                    
                    nx.draw_networkx_nodes(self.graph, pos, nodelist=[x for x, y in self.graph.nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] == key],
                                       node_color=value[0], node_shape=value[1], cmap=new_cmp, vmin=0, vmax=1)

                else:
                    nx.draw_networkx_nodes(self.graph, pos, nodelist=[x for x, y in self.graph.nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] == key],
                                       node_color=value[0], node_shape=value[1], cmap=new_cmp, vmin=0, vmax=1)
                    
                nx.draw_networkx_nodes(self.graph, pos, nodelist=[x for x, y in self.graph.nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] not in type_map.keys()],
                                       node_color='red', node_shape='X')

        if node_attribute_label:
            node_labels = nx.get_node_attributes(self.graph, node_attribute_label)
            nx.draw_networkx_labels(self.graph, pos, font_size=7, labels=node_labels)

        else:
            nx.draw_networkx_labels(self.graph, pos, font_size=7)

        connection_style = "arc3,rad=" + str(arc_rad)

        if edge_attribute:
            edge_weights = [a[edge_attribute] for _,_,a in self.graph.edges(data=True) if a and edge_attribute in a]
            if len(edge_weights) >0:
                nx.draw_networkx_edges(self.graph, pos, width=1.0, connectionstyle=connection_style, edge_cmap=new_cmp,
                                   edge_vmin=0, edge_vmax=1, edge_color=edge_weights, node_shape='s')
            else:
                nx.draw_networkx_edges(self.graph, pos, width=1.0, connectionstyle=connection_style, node_shape='s')
        else:
            nx.draw_networkx_edges(self.graph, pos, width=1.0, connectionstyle=connection_style, node_shape='s')

        if edge_attribute or node_attribute:
            plt.colorbar(sm)
        
        if pos:
            x_values, y_values = zip(*pos.values())
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * 0.25
            plt.xlim(x_min - x_margin, x_max + x_margin)
            y_max = max(y_values)
            y_min = min(y_values)
            y_margin = (y_max - y_min) * 0.25
            plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.show(block=True)
        return fig

    def serialize(self):
        return { 'reward_stats':self.reward_stats, 'depth': self.depth, "graph": nx.node_link_data(self.graph)}
          
    def save_to_file(self,directory, filename):
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        filename = f"{directory}\{filename}.json"

        data=self.serialize()
           
        print("will write graph_state object to file ", filename)
        json_str=json.dumps(data, ensure_ascii = True, cls=NpEncoder)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_str, f, ensure_ascii=False, indent=4)
            f.close()
              
        return filename
    
    def load_from_file(self,directory,filename):
        filename = f"{directory}\{filename}.json"
        with open(filename, 'r', encoding='utf-8') as fo:
            serialized_data = json.loads(json.load(fo))
            
            self.reward_stats=serialized_data['reward_stats']
            self.depth=serialized_data['depth']
            self.graph=nx.node_link_graph(serialized_data['graph'])
        return  serialized_data