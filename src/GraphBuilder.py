import random

from Tree import OpTypes, NodeTypes
import copy

from GraphState import  GraphState
from Production import Production


def choose(production_list):
    nr = random.randint(0, len(production_list) - 1)
    return production_list[nr]


class GraphBuilder:
    def __init__(self, grammar: dict[Production]):
        self.grammar = grammar

    def match(self, graph_state: GraphState):
        

        production_list = []
        for key, production in self.grammar.items():
            nodes_list = {}
            edge_list = {}

            if production.lhs_table:
                # production.printCodeList()
                has_all = True
                mappings = []
                
                for label, node in production.lhs_table.items():
                    if node.type == NodeTypes.NODE:

                        if node.value not in nodes_list.keys():

                            if "min" in node.attrib or "max" in node.attrib:

                                nodes_list[node.value] = [x for x, y in graph_state.graph.nodes(data=True) if
                                                          y['idType'] == node.value
                                                          and int(node.attrib["min"]) <= graph_state.graph.degree(
                                                              x) <= int(node.attrib["max"])]

                            else:
                                nodes_list[node.value] = [x for x, y in graph_state.graph.nodes(data=True) if
                                                          y['idType'] == node.value]

                        if bool(nodes_list[node.value]):
                            if mappings:
                                new_mappings = []
                                for n in nodes_list[node.value]:
                                    for m in mappings:
                                        if n not in m.values():
                                            k = {l:v for l,v in m.items()}
                                            k[label] = n
                                            new_mappings.append(k)
                                            
                                mappings = [ m for m in new_mappings ]

                            else:
                                for n in nodes_list[node.value]:
                                    mappings.append({label: n})
                        else:
                            has_all = False
                            mappings = []

                    elif node.type == NodeTypes.EDGE:
                        for m in mappings:
                            if node.attrib["op1"] in m and node.attrib["op2"] in m:
                                if not graph_state.graph.has_edge(m[node.attrib["op1"]], m[node.attrib["op2"]]):
                                    if m in mappings:
                                        mappings.remove(m)
                                else:
                                    edge_list[(m[node.attrib["op1"]], m[node.attrib["op2"]])]=m
                            else:
                                if m in mappings:
                                        mappings.remove(m)
                                
                #end of the loop                
                
                #remove mappings with edges that should not exist 
                to_remove=[]
                for m in mappings:
                    for src in m.values():
                        for dst in [d for d in m.values() if d!= src]:
                            if graph_state.graph.has_edge(src,dst) and (src,dst) not in edge_list:
                                to_remove.append(m)
                
                for m in to_remove:
                    if m in mappings:
                        mappings.remove(m)
                
                if has_all:
                    for m in mappings:
                        production_list.append((production, m))
                        

            elif not graph_state.graph.nodes():
                production_list.append((production, {}))

        #for p in production_list:
            #p[0].printCodeList()
            #print (f'{p[0].name} \t\t where  {p[1]}')

        return production_list

    @staticmethod
    def produce(graph_state: GraphState, p):
        # print("\t produce:----")
        # p[0].printCodeList()
        # print (" \t\t where",  p[1])

        for op in p[0].codeList:
            if op.type == OpTypes.ADDNODE:
                name = str(op.op1) + str(len([x for x, y in graph_state.graph.nodes(data=True)\
                                              if y['idType'] == str(op.op1)])+1)
                graph_state.graph.add_node(name)
                p[1][op.label] = name
                graph_state.graph.nodes()[name]['idType'] = str(op.op1)

                if graph_state.graph.nodes()[name]['idType'] is None:
                    print("CREATE ERROR")
                    graph_state.graph.nodes()[name]['idType'] = str(op.op1)

            elif op.type == OpTypes.REMNODE:
                graph_state.graph.remove_node(p[1][op.op1])

            elif op.type == OpTypes.ADDEDGE:
                n1 = p[1][op.op1]
                n2 = p[1][op.op2]

                if n1 != n2:
                    graph_state.graph.add_edge(n1, n2)

            elif op.type == OpTypes.REMEDGE:
                n1 = p[1][op.op1]
                n2 = p[1][op.op2]
                if graph_state.graph.has_edge(n1, n2):
                    graph_state.graph.remove_edge(n1, n2)

            elif op.type == OpTypes.SWAPNODE:
                n1 = p[1][op.op1]
                graph_state.graph.nodes()[n1]['idType'] = op.op2

            elif op.type == OpTypes.MERGENODE:

                n1 = p[1][op.op1]
                n2 = p[1][op.op2]
                for edge in graph_state.graph.edges(n2):
                    if edge[0] != n1 and edge[0] != n2:
                        graph_state.graph.add_edge(edge[0], n1)
                    if edge[1] != n1 and edge[1] != n2:
                        graph_state.graph.add_edge(edge[1], n1)
                graph_state.graph.nodes()[n1]['idType'] = op.op1
                graph_state.graph.remove_node(n2)
        return graph_state

    def assemble(self, graph_state: GraphState, steps: int) -> GraphState:

        for i in range(0, steps):
            # print("step ", i)
            # print(graphState.graph.nodes(data=True))
            production_list = self.match(graph_state)

            if production_list:
                self.produce(graph_state, choose(production_list))

        return graph_state
        # graphState.draw()
