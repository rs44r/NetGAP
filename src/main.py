from typing import Optional
from matplotlib.lines import segment_hits
from tqdm import tqdm

from GraphBuilder import GraphBuilder
from GraphState import GraphState
from EvaluatorLibrary import SimpleEvaluator, ValidationEvaluator
from MCSTree import MCST
from MCSTreeUtils import MCSTreeGraph
from GrammarLoader import GrammarLoader


  
class NetGapApplication:
    def __init__(self, graph_builder, graph_evaluator):
        self.graph_builder = graph_builder
        self.graph_evaluator = graph_evaluator

    def generate_single(self, num_epochs: int, iteration_limit: Optional[int] = None, time_limit: Optional[int] = None,
                        transposition_check: bool = True):

        state : GraphState = GraphState(graph_builder=self.graph_builder, graph_evaluator=self.graph_evaluator, graph=None)
        monte_carlo_tree = MCST(iteration_limit=iteration_limit, time_limit=time_limit,
                                transposition_check=transposition_check)
        tree_graph = MCSTreeGraph(None)
        path = []

        for i in (pbar := tqdm(range(0, num_epochs))):
            # print (f"starting epoch {i}")
            selected_action, tree_root = monte_carlo_tree.search(state)

            previous_state=state
            state = state.take_action(selected_action)

            if len( [x for x, y in state.graph.nodes(data=True) if y['idType'] == "M"
                        and state.graph.degree(x) >2 ]) >=1:
                print(selected_action[0].name,selected_action[1])
                previous_state.draw()
                break

            pbar.set_postfix({'visits': tree_root.num_visits, 'nodes': len(monte_carlo_tree.node_dict)})
            tree_graph.insert_nodes(tree_root)
            path.append(hash(state))
            # print(len(monte_carlo_tree.node_dict))

            if state.is_terminal():
                break
#            tree_graph.draw(path=path)
        
        tree_graph.draw(path=path)
        # print(len(tree_graph.tree.nodes()))
        state.get_reward()
        print(state.reward_stats)
         # icons = {"S": "media/switch.png", "M": "media/server.png"}
        state.draw({'S': ['#ECA47B', 's'], 'M': ['#7B9AEC', 'o']},size=(8,8),arc_rad=.125)


#     Mtree=MonteCarloTree()
#     Mtree.insert_nodes(root)
#     print(Mtree.tree.number_of_nodes())
#     Mtree.draw(path=None,y_off=35)
#     plt.show(block=True)
        return state



def main():
    grammar_file = "grammars/grammar4.txt"
    grammar_loader = GrammarLoader(grammar_file)
    grammar = grammar_loader.load()
    # grammar_loader.printProduction("r4")

    graph_builder = GraphBuilder(grammar)
    graph_evaluator =   ValidationEvaluator(module_threshold=22, segment_threshold=1) # SimpleEvaluator()

    netgap = NetGapApplication(graph_builder, graph_evaluator)

    for _ in range(0,1):
        netgap.generate_single(num_epochs=60, iteration_limit=None, time_limit=15000)


if __name__ == '__main__':
    main()
