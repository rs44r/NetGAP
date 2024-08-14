import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from networkx.drawing.nx_pydot import graphviz_layout


import json
import datetime

class SaverUtil:
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
        
    def __init__(self,directory):
        self.directory=directory
        
    def save(self,platform_list, filename=None,threshold=0.0):
        
        if filename is None:
            now = datetime.datetime.now()
            print(now.year, now.month, now.day, now.hour, now.minute, now.second)
            date= f"{now.month}{now.day}{now.hour}{now.minute}"
            filename = f"{self.directory}\dataset{date}.json"
        else:
            filename = f"{self.directory}\{filename}.json"


        data_list=[]
        for p in platform_list:
            if p.reward_stats["score"]>threshold:
                data_list.append(p.serialize())
            else:
                break
        print("will write:", len(data_list), " platforms to file ", filename)
        json_str=json.dumps(data_list, ensure_ascii = True, cls=self.NpEncoder)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_str, f, ensure_ascii=False, indent=4)
            f.close()
              
        return filename
    
        
    def load(self,filename):
        data_list=[]
        with open(filename, 'r', encoding='utf-8') as fo:
            data_list = json.loads(json.load(fo))
        
        print("loaded:", len(data_list), " platforms from file ", filename)
        return data_list

def plot_platform_stats(data):
 
    d = [v.reward_stats for v in data]
    
    latency= [y["latency_score"] for y in d]
    connectivity =[x["connectivity_score"] for x in d]
    total_cost = [c["total_cost"] for c in d]
    cost = [c["cost_score"] for c in d]
    score = [x["score"] for x in d]

    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    sc2 = ax2.scatter(latency, cost,c=total_cost, cmap='viridis')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.xlabel("latencyScore", labelpad=15)
    plt.ylabel("costScore",labelpad=15)
    
    #handles2, labels2 = sc2.legend_elements(prop="sizes", alpha=0.6)
    #legend2 = ax2.legend(handles2, labels2, loc="upper right", title="Overloads")
    cbar2 = plt.colorbar(sc2,ax=ax2)
    
    cbar2.set_label('totalCost', rotation=270, labelpad=15)
    
    plt.show()
    
  
    
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    
    sc3 = ax3.scatter(score,latency, label="latency" )
    sc4 = ax3.scatter(score,connectivity, label="connectivity")
    sc5 = ax3.scatter(score,cost, label="cost")

    
    plt.xlabel("score", labelpad=15)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.grid()
    ax3.legend()

    plt.show()
    

def drawMultipleGraphs(graphs,max_rows=8, type_map=None, node_attribute=None, edge_attribute=None,
                    size=(4, 5), arc_rad=0):
    
    
    ncols = min(4,len(graphs))
    nrows = min(max_rows, int(len(graphs) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 5))
    ax = axes.flatten()
       
    pos = []
    for j in range(min(len(graphs),ncols*nrows)):
        pos.append(graphviz_layout(graphs[j], prog="neato"))  # 'sfdp' 'fdp' 'neato'
    
    
    top = cm.get_cmap('summer', 256)
    bottom = cm.get_cmap('autumn', 256)
    
    new_colors = np.vstack((top(np.linspace(0, 0.8, 6)),
                                bottom(np.linspace(1, 0, 4))))
    new_cmp = ListedColormap(new_colors, name='RedGreen')
    
    sm = plt.cm.ScalarMappable(cmap=new_cmp, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(np.array([]))

    for i in range(min(len(graphs),ncols*nrows)):

        
        ax[i].set_title("Graph {}".format(i))
        ax[i].set_axis_off()

        #drawing the nodes here
        
        if type_map and not node_attribute:
            # get the typemap and draws from typemap
            for key, value in type_map.items():
                nx.draw_networkx_nodes(graphs[i], pos[i], nodelist=
                [x for x, y in graphs[i].nodes(data=True) if 'idType' in y and y['idType'] == key],
                                       node_color=value[0], node_shape=value[1], ax=ax[i])

            nx.draw_networkx_nodes(graphs[i], pos[i], nodelist=
            [x for x, y in graphs[i].nodes(data=True) if 'idType' in y and y['idType'] not in type_map.keys()],
                                   node_color='red', node_shape='X', ax=ax[i])
        
        elif type_map and node_attribute:

            for key, value in type_map.items():
                if value[0] == 'att':
                    nodes, node_values = zip(*nx.get_node_attributes(graphs[i], node_attribute).items())
                    value[0] = node_values
                nx.draw_networkx_nodes(graphs[i], pos, nodelist=[x for x, y in graphs[i].nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] == key],
                                       node_color=value[0], node_shape=value[1], cmap=new_cmp, vmin=0, vmax=1, ax=ax[i])

                nx.draw_networkx_nodes(graphs[i], pos, nodelist=[x for x, y in graphs[i].nodes(data=True)
                                                                  if y and 'idType' in y and y['idType'] not in type_map.keys()],
                                       node_color='red', node_shape='X', ax=ax[i])
                                       
        elif not typemap and node_attribute:
            # get the attributes and draws it
            node_att = nx.get_node_attributes(graphs[i], node_attribute)
            nx.draw_networkx_nodes(graphs[i],  pos[i], cmap=newcmp, vmin=0, vmax=1, node_color=node_att, ax=ax[i])
            
        elif not typemap and not node_attribute:
            nx.draw_networkx_nodes(graphs[i],  pos[i], ax=ax[i])
        
        
        #drawing node labels here
        nx.draw_networkx_labels(graphs[i],  pos[i], font_size=7, ax=ax[i])
        
        
        #drawing edges here

        connection_style = "arc3,rad=" + str(arc_rad)
        
        if edge_attribute:
            edges, edge_weights = zip(*nx.get_edge_attributes(graphs[i], edge_attribute).items())
            nx.draw_networkx_edges(graphs[i],  pos[i], width=1.0, connectionstyle=connection_style, edge_cmap=new_cmp,
                                   edge_vmin=0, edge_vmax=1, edge_color=edge_weights, node_shape='s',ax=ax[i])
        else:
            nx.draw_networkx_edges(graphs[i],  pos[i], width=1.0, connectionstyle=connection_style, node_shape='s')
        
        
        #drawing colorbar
       # if edge_attribute or node_attribute:
          #  plt.colorbar(sm)
        
    plt.show(block=True)

def drawSingleGraph(graph, typemap=None, node_attribute_label=None, node_attribute=None, edge_attribute=None,
                    size=(4, 6), arc_rad=0):
                    

    top = cm.get_cmap('summer', 256)
    bottom = cm.get_cmap('autumn', 256)
    newcolors = np.vstack((top(np.linspace(0, 0.8, 6)),
                           bottom(np.linspace(1, 0, 4))))
    newcmp = ListedColormap(newcolors, name='RedGreen')

    sm = plt.cm.ScalarMappable(cmap=newcmp, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array(np.array([]))

    pos = graphviz_layout(graph, prog="circo")
    pos_offset = {}

    for k, v in pos.items():
        pos_offset[k] = (v[0] * 0.75 + 12, v[1] * 0.75 - 12)

    if not typemap and not node_attribute:
        nx.draw_networkx_nodes(graph, pos)

    elif typemap and not node_attribute:
        for key, value in typemap.items():
            nx.draw_networkx_nodes(graph, pos, nodelist=
            [x for x, y in graph.nodes(data=True) if 'idType' in y and y['idType'] == key],
                                   node_color=value[0], node_shape=value[1])

        nx.draw_networkx_nodes(graph, pos, nodelist=
                                [x for x, y in graph.nodes(data=True) if 'idType' in y and y['idType'] not in typemap.keys()],
                               node_color='red', node_shape='X')

    elif not typemap and node_attribute:
        node_att = nx.get_node_attributes(graph, node_attribute)
        nx.draw_networkx_nodes(graph, pos, cmap=newcmp, vmin=0, vmax=1, node_color=node_att)

    elif typemap and node_attribute:

        for key, value in typemap.items():
            if value[0] == 'map':
                nodes, node_values = zip(*nx.get_node_attributes(graph, node_attribute).items())
                value[0] = node_values
            nx.draw_networkx_nodes(graph, pos, nodelist=
                                                        [x for x, y in graph.nodes(data=True) if 'idType' in y and y['idType'] == key],
                                   node_color=value[0], node_shape=value[1], cmap=newcmp, vmin=0, vmax=1)

            nx.draw_networkx_nodes(graph, pos, nodelist=
            [x for x, y in graph.nodes(data=True) if 'idType' in y and y['idType'] not in typemap.keys()],
                                   node_color='red', node_shape='X')

    if node_attribute_label:
        node_labels = nx.get_node_attributes(graph, node_attribute_label)
        nx.draw_networkx_labels(graph, pos, font_size=8, labels=node_labels)

    else:
        nx.draw_networkx_labels(graph, pos, font_size=8)

    connectionstyle = "arc3,rad=" + str(arc_rad)

    if edge_attribute:
        edges, edge_weights = zip(*nx.get_edge_attributes(graph, edge_attribute).items())
        nx.draw_networkx_edges(graph, pos, width=1.0, connectionstyle=connectionstyle, edge_cmap=newcmp, edge_vmin=0,
                               edge_vmax=1, edge_color=edge_weights, node_shape='s')
    else:
        nx.draw_networkx_edges(graph, pos, width=1.0, connectionstyle=connectionstyle, node_shape='s')

    if edge_attribute or node_attribute:
        plt.colorbar(sm)

    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    y_max = max(y_values)
    y_min = min(y_values)
    y_margin = (y_max - y_min) * 0.25
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.show(block=False)

