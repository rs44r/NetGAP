from typing import Optional

from networkx.drawing.nx_pydot import graphviz_layout
from MCSTree import MonteCarloTreeNode
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import datetime


class MCSTreeGraph:
    def __init__(self, root: Optional[MonteCarloTreeNode]):
        self.root: MonteCarloTreeNode = root
        self.tree = nx.DiGraph()

        if root is not None:
            self.insert(self.tree, self.root)
            print(self.tree.number_of_nodes())


    def draw(self, path=None, y_off=0.0, fig_size=(8,8),save_filename=None,save_extension='.svg'):
        
        self.tree.remove_edges_from(nx.selfloop_edges(self.tree))
        values = []
        # print(self.tree.nodes(data=True))
        
        max_v=0
        for node, attributes in self.tree.nodes(data=True):
            node_value = attributes['total_reward']/attributes['num_visits']
            values.append(node_value)
            max_v=max(max_v,node_value)
        
        # print(values)
        #values=[v/max_v for v in values]
        values = np.clip(np.array(values), a_min=0.1, a_max=None)
        # values = np.array(values) / np.sqrt(np.sum(np.array(values) ** 1))
        # print(values)
        fig=plt.figure(figsize=fig_size)
        # plt.title(max_value)
        #pos_off = {}
        pos = graphviz_layout(self.tree, prog='dot')
        #for k, v in pos.items():
            #pos_off[k] = (v[0], v[1] + y_off)
        # node_labels = nx.get_node_attributes(self.graph,'idType')
        # nx.draw_networkx_labels(self.tree, pos_off, font_size=8)
        nx.draw_networkx_edges(self.tree, pos, width=1.0, alpha=0.5)
        
        cbar=nx.draw_networkx_nodes(self.tree, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=32 * values)
        
        if path is not None:
            path_sane = [id(p) for p in path if id(p) in self.tree.nodes()]
            nx.draw_networkx_nodes(self.tree, pos, nodelist=path_sane, node_shape='+', node_size=64 ,node_color='red', alpha=0.5)
        
        plt.colorbar(cbar,location="top",shrink=0.6,aspect=40,pad=0.01)
       
        
        if save_filename is not None:
            now = datetime.datetime.now()
            date= f"{now.month}{now.day}{now.hour}{now.minute}"
            save_filename += f"_size{self.tree.number_of_nodes()}_{date}"
            save_filename += save_extension
            plt.savefig(save_filename)
            
        plt.show(block=True)

    # def insert(self, tree, root):
    #     tree.add_node(id(root), total_reward=root.total_reward)
    #     for action, children in root.children.values():
    #         if children not in tree:
    #             self.insert(tree, children)
    #         tree.add_edge(id(root), id(children))
    #
    def insert(self, tree, root):

        if root not in tree.nodes():
            tree.add_node(id(root), total_reward=root.total_reward, num_visits=root.num_visits)
        for action, children in root.children.values():
            if id(children) not in tree.nodes():
                self.insert(tree, children)
            tree.add_edge(id(root), id(children))
       

    def insert_nodes(self, root):
        self.insert(self.tree, root)
        print(self.tree.number_of_nodes())
