# import math
import math
from typing import Tuple, Dict, List
import json
import networkx as nx
import numpy as np
from GraphState import GraphEvaluator, GraphState
from GeneticAlgorithm import GeneticAlgorithm, Individual, ExponentialRankSelector, PartiallyMappedCrossover, ReverseSequenceMutation
# import mypy

from GeneticAlgorithm import Individual, GeneticAlgorithm, ExponentialRankSelector, ReverseSequenceMutation, PartiallyMappedCrossover
from typing import Dict, List
import json
import math

class OptimizerEvaluator():
    
    def __init__(self, data_flows, graph=None):
        self.data_flows=data_flows        
        self.graph=graph
        
        if graph is not None:
             self.module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
            
        self.shortest_paths = {}
        self.simple_paths = {}
        self.switch_paths = {}
        
    def calculate_SW_connectivity(self,graph,cutoff=3):
        
        switch_connectivity=0
        switch_list=list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'S'])
        
        count=0
        
        for s in switch_list:
            leaves = [d for d in switch_list if d!=s and (s,d) not in self.switch_paths]
            count+=len(leaves )
            if len(leaves)>0:
                paths = nx.all_simple_paths(graph, s, leaves, cutoff=6)
                for path in paths:
                    path=list(path)
                    if (s, path[-1]) in self.switch_paths:
                        self.switch_paths[(s,path[-1])]+=1
                        n_paths=self.switch_paths[(s,path[-1])]
                        if n_paths==2:
                            switch_connectivity+=1
                    else: 
                        self.switch_paths[(s,path[-1])]=1    
                                              
        return len([n for n in self.switch_paths.values() if n>1])/max(1,len(list(self.switch_paths.values())))
        
    
    def calculate_HP_connectivity(self,graph,module_list, cutoff=3):
           
        HP_connectivity=0
        
        HP_flow_list = [c for c in self.data_flows if c[3] == 'high']
        HP_flow_count = len(HP_flow_list)
        
        for c in HP_flow_list:
            switch_sender=[n for n in graph.neighbors(module_list[c[0]])][0]
            switch_receiver= [n for n in graph.neighbors(module_list[c[1]])][0]
            if(switch_sender==switch_receiver):
                pass
            else:
                if (switch_sender, switch_receiver) not in self.switch_paths:
                    n_paths = len(list(nx.all_simple_paths(graph, switch_sender, switch_receiver,cutoff=cutoff)))
                    self.switch_paths[(switch_sender, switch_receiver)]=n_paths
                else: 
                    n_paths = self.switch_paths[(switch_sender, switch_receiver)]
                    
                if n_paths> 1:
                    HP_connectivity+=1
                
        return HP_connectivity/HP_flow_count
            

    def count_hops(self,graph,module_list):
        
        hops=np.zeros(len(self.data_flows))
        for index, c in enumerate(self.data_flows):
            if (module_list[c[0]], module_list[c[1]]) not in self.shortest_paths:
                shortest_path = list(nx.shortest_path(graph, source=module_list[c[0]], target=module_list[c[1]])) 
                self.shortest_paths[(module_list[c[0]], module_list[c[1]])]=shortest_path
            else:
                shortest_path=self.shortest_paths[(module_list[c[0]], module_list[c[1]])]
            hops[index] = len(shortest_path)-1
                
        return hops
    
    def calculate_link_loads(self,graph,module_list,edge_list):
        
        edge_loads = np.zeros(len(edge_list))
    
        for index, c in enumerate(self.data_flows):
            if (module_list[c[0]], module_list[c[1]]) not in self.shortest_paths:
                shortest_path = list(nx.shortest_path(graph, source=module_list[c[0]], target=module_list[c[1]])) 
                self.shortest_paths[(module_list[c[0]], module_list[c[1]])]=shortest_path
            else:
                shortest_path=self.shortest_paths[(module_list[c[0]], module_list[c[1]])]
            for h in range(0, len(shortest_path) - 1):
                edge_loads[edge_list.index((shortest_path[h], shortest_path[h + 1]))] += c[2]
        
        return edge_loads
        
    def count_link_overloads(self,loads,target):
        return (loads / 100e6 > target).sum()
    
    def max_link_load(self,loads):
        return np.max(loads / 100e6)
    
        
    def calculate_LC_scores(self, graph, module_list, alpha=0.4):
        max_load_target=0.79999
        
        edge_list = list(graph.edges())
        
        loads = self.calculate_link_loads(graph,module_list,edge_list)
        hops = self.count_hops(graph,module_list)
              
        max_load = self.max_link_load(loads)
        overload_count = self.count_link_overloads(loads,max_load_target)
        mean_hops = np.mean(hops)
        max_hops = np.max(hops).astype(int)
        scaling_hops = math.floor(mean_hops)
        scaling_hops=0
        
        latency_score = 2*math.exp(1-max_load/max_load_target   - overload_count)/mean_hops
        
        SW_connectivity = self.calculate_SW_connectivity(graph,cutoff=max_hops+2)
        HP_connectivity = self.calculate_HP_connectivity(graph,cutoff=max_hops+2, module_list=module_list)
        
        connectivity_score = alpha*SW_connectivity + (1-alpha)*HP_connectivity
        
        for i,e in enumerate(edge_list):
            graph.edges[e]["utilization"] = loads[i] / 100e6
        
        return latency_score, connectivity_score, \
                                {'max_load': max_load, 'overload_count':overload_count, 'mean_hops': mean_hops,\
                               'max_hops': max_hops, 'latency_score':latency_score,'connectivity_score':connectivity_score,\
                               'SW_connectivity':SW_connectivity, 'HP_connectivity': HP_connectivity}
    
    def evaluate(self, individual : Individual):
        
        if isinstance(individual,Individual): 
            ordering_list = individual.genome 
            if self.graph is not None:
                module_order=[self.module_list[n] for n in ordering_list] 
                latency, connectivity, stats=self.calculate_LC_scores(self.graph,module_order)
                beta=0.25
                return (latency*(1-beta)+connectivity*(beta))
            else:
                raise Exception("Evaluate can only be used if graph is set")
        else:
                raise TypeError("Evaluate argument has to be a genetic algorithm individual")
   




class GeneticEvaluator(GraphEvaluator):
    module_threshold : int
    generations : int
    weights : Dict[str,float]
    data_flows : List[tuple[int,int,float]]


    def __init__(self, weights : Dict[str,float], module_threshold: int =0, pop_size : int =50, generations : int = 3, filename : str ="data/connections.json",seed=12345):
        self.pop_size = pop_size 
        self.generations = generations 
        self.module_threshold=module_threshold
        self.weights = weights
        self.rng = np.random.default_rng(seed)
      
        

        with open(filename) as file:
            self.data_flows = json.loads(json.load(file))
            
        self.module_threshold=max([max(f[0],f[1]) for f in self.data_flows])+1
    
    def update_seed(self,seed):
        self.rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        ret_str = f"module_threshold: {self.module_threshold}\n"
        ret_str += f"pop_size: {self.pop_size} \n"
        ret_str += f"generations {self.generations} \n"
        ret_str += str(self.weights)

        return ret_str


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
    
   

    def evaluate_LC(self, graph,alpha=0.5):

        module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
        
        genome_size = len(module_list)
        
        latency_evaluator=OptimizerEvaluator(self.data_flows,graph=graph)
        
        
        if len(module_list)<self.module_threshold:
            SW_connectivity=latency_evaluator.calculate_SW_connectivity(graph,6)
            connectivity_score=SW_connectivity*(alpha)
            #connectivity_score=0
            return 0, connectivity_score, \
                    {'connectivity_score': connectivity_score,'SW_connectivity':SW_connectivity,'latency_score':0}, \
                    module_list
        
        #SW_connectivity=latency_evaluator.calculate_SW_connectivity(graph,6)
        
        seed_sel=self.rng.integers(low=1,high=np.iinfo(np.int32).max)
        seed_cross=self.rng.integers(low=1,high=np.iinfo(np.int32).max)
        seed_mut=self.rng.integers(low=1,high=np.iinfo(np.int32).max)
        
        selector=ExponentialRankSelector(seed=seed_sel)
        crossop=PartiallyMappedCrossover(seed=seed_cross)
        mutop=ReverseSequenceMutation(seed=seed_mut)
        
              
        class GenomeInitializer:
            def __init__(self, seed=None):
                self.rng=np.random.default_rng(seed)
            def __call__(self,genome):
                linspace=np.linspace(start=0,stop=len(genome)-1, num=len(genome),dtype=int)
                self.rng.shuffle(linspace)
                return linspace
        
        seed_init=self.rng.integers(low=1,high=np.iinfo(np.int32).max)
        init_fn=GenomeInitializer(seed_init)
        
        ga = GeneticAlgorithm(genome_size = genome_size, crossover_operator=crossop, 
                          mutation_operator=mutop, evaluator=latency_evaluator, selector=selector)

        individuals, fitness= ga.run(pop_size = self.pop_size, generations=self.generations, init_fn=init_fn)
        
    
        module_list=[module_list[n] for n in individuals[0].genome]
        
        latency_score,connectivity_score,stats=latency_evaluator.calculate_LC_scores(graph,module_list,alpha=alpha )
        
        return latency_score, connectivity_score, stats, module_list
    
    def evaluate_cost(self,graph):
        module_cost : float =10
        link_cost : float =0.1
        
        module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
       
        total_cost  = (graph.number_of_nodes() * 10 + graph.number_of_edges() * .1)
        normalized_cost = ((self.module_threshold+1)*module_cost + self.module_threshold*link_cost) / max(1,total_cost)
        
        if normalized_cost > 1:
            return 1/normalized_cost, {'total_cost':total_cost,'cost_score':1/normalized_cost}
        else:
            return normalized_cost, {'total_cost':total_cost,'cost_score':normalized_cost}
    
    
    def evaluate(self,state):
  
        self.trim_graph(state.graph)
        
        graph=state.graph
        
        stats={}
         
        latency_score, connectivity_score, LC_stats, module_list = self.evaluate_LC(graph,alpha=0.3)
        
        cost_score, cost_stats = self.evaluate_cost(graph)
        
        
        
        
        module_ratio = len(module_list)/self.module_threshold  
        module_score = min(module_ratio,1) #if module_ratio<=1 else -(4*((module_ratio-1)**0.5))
            
        stats.update(LC_stats),stats.update(cost_stats),stats.update({'module_score': module_score})
        
        score=0
        score += latency_score*self.weights['latency'] + connectivity_score*self.weights['connectivity'] \
                + cost_score*self.weights['cost'] + module_score*self.weights['modules']
        score = score / (self.weights['latency']+self.weights['connectivity']+self.weights['cost']+self.weights['modules'])
        score = max(5e-3,score)
        #score = score**2
        stats.update({'score':score})
        
        return score, stats
        
    def is_terminal(self, state) -> bool:
        #return False
        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) >= self.module_threshold
        #cost = state.graph.number_of_nodes() > 28
        
        #number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) > self.module_threshold
        return number_of_modules #| cost
