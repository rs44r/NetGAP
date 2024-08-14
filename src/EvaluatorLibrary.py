# import math
import math
from itertools import combinations
from statistics import mean
from typing import Tuple
import json
import networkx as nx
import numpy as np
import random
from GraphState import GraphEvaluator, GraphState
from GeneticAlgorithm import GeneticAlgorithm, Individual, ExponentialRankSelector, PartiallyMappedCrossover, ReverseSequenceMutation
# import mypy

class SimpleEvaluator(GraphEvaluator):
    def find_longest_path(self, state: GraphState):
        module_list = list([x for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M'])
        try:
            max_path_length = mean(
                [nx.shortest_path_length(state.graph, pair[0], pair[1]) for pair in combinations(module_list, 2)])
        except ValueError:

            # print("Value Error Exception")
            return math.inf
        return max_path_length

    def evaluate(self, state: GraphState) -> Tuple[float, dict]:
        cost = state.graph.number_of_nodes() * 10 + state.graph.number_of_edges() * .1
        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M'])
        reward = math.exp(-(number_of_modules/20-1)) / cost
        max_path_length = math.inf
        if number_of_modules > 0:
            # print("will look for path length")
            max_path_length = self.find_longest_path(state)
            reward =math.exp(-(number_of_modules/20-1))  / (max_path_length*2 )/ cost
            # print("found path length ")
        return reward, {'reward': reward, 'cost': cost, 'n_mods': number_of_modules, 'max_path_length': max_path_length}

    def is_terminal(self, state: GraphState) -> bool:

        total_cost = (state.graph.number_of_nodes() * 10 + state.graph.number_of_edges() * .1) > 280
        number_of_modules = sum(
            [1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) == 20
        return number_of_modules | total_cost


class ValidationEvaluator():

    def __init__(self, module_threshold=8, segment_threshold=2):
        self.moduleThreshold = module_threshold
        self.segmentThreshold = segment_threshold
        self.score = {}
        self.TopState = None
        self.connections=[]
            
        with open("data/connections.json") as file:
            self.connections = json.loads(json.load(file))
    
    def is_terminal(self, state) -> bool:
        #return False
        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) == self.moduleThreshold
        return number_of_modules 

    def trim_graph(self, graph):
        graph
        nodes_to_remove=[]
        for x,y in graph.nodes(data=True):
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

    def evaluate_connectivity_load_opt(self, graph, connections, it=25):
        
        best_connectivity = 0.0
        best_score = 0.0
        best_hops = np.ones(len(connections)) * np.inf
        best_violations = 0.0
        high_count = np.inf
        overload_count = 0.0
        max_load = 0.0
        connectivity_score = 0.0
        edge_list = list(graph.edges())
        best_load = np.ones(len(edge_list)) * np.inf
        module_list= list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
        switch_list=list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'S'])

        if len(module_list) >= 22:
            self.trim_graph(graph)
            switch_list=list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'S'])
            edge_list = list(graph.edges())
            best_load = np.ones(len(edge_list)) * np.inf
            #shortest_path_dict={}
            #for pair in permutations(all_module_list):
                #shortest_path_dict[pair] = list(nx.shortest_path(graph, source=pair[0], target=pair[1]))

            global shortest_path

            for i in range(0, it):

                current_load = np.zeros(len(edge_list))
                current_hops = np.zeros(len(connections))
                current_connectivity = 0
                current_violations = 0

                random.shuffle(module_list)

                for index, c in enumerate(connections):

                    module_list = module_list

                    # for all, adds load on the edges
                    shortest_path = list(nx.shortest_path(graph, source=module_list[c[0]], target=module_list[c[1]]))  # this line could be done just1once and a map created as before
                    #shortest_path=shortest_path_dict[(module_list[c[0] - offset],module_list[c[1] - offset])]
                    current_hops[index] = len(shortest_path) - 1

                    for h in range(0, len(shortest_path) - 1):
                        current_load[edge_list.index((shortest_path[h], shortest_path[h + 1]))] += c[2]

                max_load = np.max(current_load / 100e6)/0.79
                overload_count = (best_load / 100e6 > 0.79).sum()-1

                #if (max_load < 1.0): #or force_res:
                if overload_count <=1:
                    
                    high_list = [c for c in connections if c[3] == 'high']
                    high_count = len(high_list)
                    for c in high_list:
                        if len(list(nx.all_simple_paths(graph, module_list[c[0]], module_list[c[1]],cutoff=np.max(best_hops))))> 1:
                            current_connectivity+=1
                    
               # if high_count and high_count > 0 and max_load > 0:
                    #current_score = 1 / (max_load ** overload_count) #* current_connectivity / high_count
                if max_load > 1:
                    current_score = 1 / (max_load**overload_count)
                elif max_load > 0:
                    current_score = 1 / (max_load*(overload_count+1))*current_connectivity / high_count
                else:
                    current_score = 0.0

                if current_score > best_score:
                    best_connectivity = current_connectivity
                    best_load = current_load
                    best_hops = current_hops
                    best_score = current_score
                    best_violations = current_violations
        
            overload_count = (best_load / 100e6 > 0.79).sum()-1
            max_load = np.max(best_load / 100e6) /0.79
            connectivity_score = best_connectivity / high_count*0.5

        else:
            overload_count=10
            best_hops=1  

        switch_connectivity=0
        for m in switch_list:
            for n in switch_list:
                if m != n:
                    if len(list(nx.all_simple_paths(graph, m, n, cutoff=3 )))> 1:
                            switch_connectivity+=1
        connectivity_score+=0.5*switch_connectivity/max(len(switch_list)**2,1)


        for i in range(0, len(edge_list)):
            e = graph.edges[edge_list[i]]
            e["utilization"] = best_load[i] / 100e6

        #print( connectivity_score, best_connectivity, max_load, overload_count, np.mean(best_hops)) 
        return connectivity_score, best_connectivity, max_load,max(0, overload_count), np.mean(best_hops), best_violations

    @staticmethod
    def compute_score(score):
        # CalculateScore
        temp_score = 0

        alpha = 12
        beta = 3
        gamma = 6
        delta = 5
        eps = 12
        min_hops=3

        temp_latency_score = math.exp(min_hops)*math.exp(-score["loadScore"] *score["meanHops"]  -score["overloadCount"])


        score["segment_score"]  = score["segment_ratio"]  * math.exp(-score["segment_violations"])

        temp_score += score["connectivityScore"] * alpha

        if score["segment_ratio"]  >= 1:
            temp_score += beta * score["segment_score"]

        temp_score += score["moduleScore"] * gamma
        temp_score += score["costNormalized"] * delta
        # temp_score+=math.exp(-score.loadScore-score.overloadCount)*eps
        temp_score += temp_latency_score * eps
        temp_score = temp_score / (alpha + beta + gamma + delta + eps)

        return temp_score, temp_latency_score

    def evaluate(self, state: GraphState, opt_steps=50):
        
        if self.TopState is None:
            self.TopState = state

        graph = state.graph

        self.score = {}

        module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
       # module_Y_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'MY'])
        #module_X_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'MX'])

        # HERE STARTS THE NETWORKING PART

        # connections=connections_low

        # HERE STARTS THE SEGMENTS PART
        self.score["number_of_segments"] = 1 + len(
            list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'B']))
        self.score["segment_ratio"] = self.score["number_of_segments"] / self.segmentThreshold

        if len(self.connections) > 0:
            self.score["connectivityScore"] , self.score["connectivityCount"] , self.score["loadScore"] , self.score["overloadCount"] , \
            self.score["meanHops"], self.score["segment_violations"]  = \
                self.evaluate_connectivity_load_opt(graph, self.connections,it=250)
        # HERE STARTS THE COSTS PART
        self.score["totalCost"]  = (graph.number_of_nodes() * 10 + graph.number_of_edges() * .1)
        self.score["costNormalized"] = ((self.moduleThreshold + 2 * self.segmentThreshold - 1) * 10 + (
                self.moduleThreshold + self.segmentThreshold) * .1) / self.score["totalCost"]

        # HERE STARTS THE MODULES PART

        self.score["number_of_modules"]  = len(module_list)
        self.score["moduleScore"]  = min(1.0, len(module_list) / self.moduleThreshold)

        # HERE CALCULATES SCORE
        self.score["score"], self.score["latencyScore"] = self.compute_score(self.score)



        # IDENTIFY TOP SCORE and if top REFINED OPT
        # print("{:.4}".format(state.score.score))



        return self.score["score"], self.score
        
class LatencyEvaluator():
    
    def __init__(self, data_flows, graph=None):
        self.data_flows=data_flows        
        self.graph=graph
        
        if graph is not None:
             self.module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
            
        self.shortest_paths = {}
            

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
        
    def count_link_overloads(self,loads):
        return (loads / 100e6 > 0.799).sum()
    
    def max_link_load(self,loads):
        return np.max(loads / 100e6)/0.79  
    
        
    def calculate_latency_score(self, graph, module_list):
        
        edge_list = list(graph.edges())
        
        loads = self.calculate_link_loads(graph,module_list,edge_list)
        hops = self.count_hops(graph,module_list)
        
            
        max_load = self.max_link_load(loads)
        overload_count = self.count_link_overloads(loads)
        mean_hops = np.mean(hops)
        max_hops = np.max(hops)
        scaling_hops = math.floor(mean_hops)
        
        latency_score = math.exp(scaling_hops)*math.exp(-max_load*mean_hops - overload_count)
        
        for i,e in enumerate(edge_list):
            graph.edges[e]["utilization"] = loads[i] / 100e6
        
        return latency_score, {'max_load': max_load, 'overload_count':overload_count, 'mean_hops': mean_hops,\
                               'max_hops': max_hops, 'latency_score':latency_score}
    
    def evaluate(self, individual : Individual):
        
        if isinstance(individual,Individual):
            ordering_list = individual.genome
            if self.graph is not None:
                module_order=[self.module_list[n] for n in ordering_list] 
                score,stats=self.calculate_latency_score(self.graph,module_order)
                return score
            else:
                raise Exception("Evaluate can only be used if graph is set")
        else:
                raise TypeError("Evaluate argument has to be a genetic algorithm individual")
       
    
    
       
class GeneticEvaluator(GraphEvaluator):

    def __init__(self, module_threshold=8, segment_threshold=2, pop_size=50, generations=3, filename="data/connections.json"):
        self.module_threshold = module_threshold
        self.segment_threshold = segment_threshold
        self.pop_size = pop_size 
        self.generations = generations 
        
        
        
        with open(filename) as file:
            self.data_flows = json.loads(json.load(file))
        
    
    def is_terminal(self, state) -> bool:
        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) == self.moduleThreshold
        return number_of_modules 

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
        
    
   
           
    def calculate_HP_connectivity(self,graph,module_list, cutoff=3):
    
        if len(module_list)<self.module_threshold:
            return 0
        
        HP_connectivity=0
        
        HP_flow_list = [c for c in self.data_flows if c[3] == 'high']
        HP_flow_count = len(HP_flow_list)
        
        for c in HP_flow_list:
            if len(list(nx.all_simple_paths(graph, module_list[c[0]], module_list[c[1]],cutoff=cutoff)))> 1:
                HP_connectivity+=1
        return HP_connectivity/HP_flow_count
                    
    def calculate_SW_connectivity(self,graph,cutoff=3):
        switch_connectivity=0
        
        switch_list=list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'S'])
        
        for m in switch_list:
            for n in switch_list:
                if m != n:
                    if len(list(nx.all_simple_paths(graph, m, n, cutoff=cutoff )))> 1:
                            switch_connectivity+=1
        return switch_connectivity/max(len(switch_list)**2,1)
    
    def evaluate_connectivity(self, graph, cutoff=3, alpha=0.5, module_list = None):
        
        if module_list is None:
            module_list= list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
        else: 
            module_list=module_list
            
        
        HP_connectivity = self.calculate_HP_connectivity(graph,cutoff=cutoff, module_list=module_list)
        SW_connectivity = self.calculate_SW_connectivity(graph,cutoff)
        
        score = alpha*SW_connectivity + (1-alpha)*HP_connectivity
        stats={'connectivity_score':score,'SW_connectivity':SW_connectivity, \
               'HP_connectivity': HP_connectivity}
        
        return score, stats

    def evaluate_latency(self, graph):

        module_list = list([x for x, y in graph.nodes(data=True) if 'idType' in y if y['idType'] == 'M'])
        
        
        if len(module_list)<self.module_threshold:
            return 0,{'max_hops': 8}, module_list
        
        genome_size = len(module_list)
        
        latency_evaluator=LatencyEvaluator(self.data_flows,graph=graph)
        selector=ExponentialRankSelector()
        crossop=PartiallyMappedCrossover()
        mutop=ReverseSequenceMutation()
        
        def init_fn(genome):
            linspace=np.linspace(start=0,stop=len(genome)-1, num=len(genome),dtype=int)
            np.random.shuffle(linspace)
            return linspace
        

        ga = GeneticAlgorithm(genome_size = genome_size, crossover_operator=crossop, \
                          mutation_operator=mutop, evaluator=latency_evaluator, selector=selector)

        individuals, fitness= ga.run(pop_size = self.pop_size, generations=self.generations, init_fn=init_fn)
        
        #print([str(i) for i in individuals[0:5]] ,fitness[0:5])
        module_list=[module_list[n] for n in individuals[0].genome]
        
        score,stats=latency_evaluator.calculate_latency_score(graph,module_list )
        
        return score, stats, module_list
    
    def evaluate_cost(self,graph):
        module_cost=10
        link_cost=0.1
        total_cost  = (graph.number_of_nodes() * 10 + graph.number_of_edges() * .1)
        normalized_cost = ((self.module_threshold + 2 * self.segment_threshold - 1) * module_cost \
                           + (self.module_threshold + self.segment_threshold+2) * link_cost) / total_cost
        return normalized_cost, {'total_cost':total_cost,'cost_score':normalized_cost}
    
    
    def evaluate(self,state):
  
        self.trim_graph(state.graph)
        
        graph=state.graph
        
        stats={}

 
        latency_score, latency_stats, module_list = self.evaluate_latency(graph)
        
        connectivity_score, connectivity_stats = self.evaluate_connectivity(graph, alpha=0.5, cutoff=latency_stats['max_hops']+2, module_list=module_list)
        
        cost_score, cost_stats = self.evaluate_cost(graph)
        
        module_score= len(module_list)/self.module_threshold  
    
            
        stats.update(latency_stats),stats.update(connectivity_stats),stats.update(cost_stats)
        
        alpha = 12
        beta = 0 #3
        gamma = 6
        delta = 5
        eps = 12
        
        score=0
        score+= latency_score*eps + connectivity_score*alpha + cost_score*delta + module_score*gamma
        score = score / (alpha + beta + gamma + delta + eps)
        
        stats.update({'score':score})
        
        return score, stats
        
    def is_terminal(self, state) -> bool:
        #return False
        number_of_modules = sum([1 for x, y in state.graph.nodes(data=True) if y and 'idType' in y if y['idType'] == 'M']) == self.module_threshold
        
        return number_of_modules 
    
