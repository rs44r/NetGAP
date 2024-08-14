from multiprocessing import Process, Pool
import numpy as np
from tqdm import tqdm 
import os
import math
import datetime
import time
import sys
import getopt


from GraphBuilder import GraphBuilder
from GraphState import GraphState, GraphEvaluator
from GeneticEvaluator import GeneticEvaluator
from MCSTree import MCST,State, MonteCarloTreeNode, Policies
from MCSTreeUtils import MCSTreeGraph
from GrammarLoader import GrammarLoader
from GraphUtils import *
from HybridEvaluator import HybridEvaluator
from ComparisonLib import NetGAPStats, ComparisonSaver
        
def app(num_samples, iteration_limit=None, time_limit=None, seed=None):
    
    
    if iteration_limit is None and time_limit is None:
        raise Exception("Either time or iteration limits must be provided")


    grammar_file = "C:/Users\smora\Documents\LiU\Workspace\\NetGAP\grammars\grammar3.txt"
    grammar_loader = GrammarLoader(grammar_file)
    grammar = grammar_loader.load()
  
    graph_builder = GraphBuilder(grammar)
    
    weights={'latency': 12, 'connectivity': 9, 'cost' : 24, 'modules': 6 }
    genetic_evaluator =  GeneticEvaluator(weights=weights, pop_size=50, generations=3, filename = "C:/Users\smora\Documents\LiU\Workspace\\NetGAP\data\connections.json",seed=seed)
    os.chdir("C:/Users\smora\Documents\LiU\Workspace\\NetGAP\models")
    neural_evaluator = HybridEvaluator( filename = "torch_model",weights=weights,seed=seed)
    
    
    rng=np.random.default_rng(seed)  
    
    history=[] 
    for i in range(num_samples):
        sample_seed=rng.integers(low=1,high=np.iinfo(np.int32).max)
        for evaluator in [genetic_evaluator,neural_evaluator]:
            
            evaluator.update_seed(sample_seed)
            tree_policies=Policies(seed=sample_seed)
            
            tree = MCST(iteration_limit=iteration_limit, time_limit=time_limit, max_depth=40, transposition_check=True, exploration_constant=12,\
                        rollout_policy= tree_policies.RandomPolicy, selection_policy= tree_policies.UCTPolicy)
        
            state = GraphState(graph_builder=graph_builder, graph_evaluator=evaluator) 
            root =  MonteCarloTreeNode(state)
            
            node=root
            start = time.perf_counter()
            for i in range(tree.max_depth):
                selected_action, node, it = tree.search(node) 
    
                if node.state.is_terminal():
                    break
            
            
            end = time.perf_counter()
            delta_time=end-start
          
            
            node.state.graph_evaluator=genetic_evaluator
            node.state.get_reward()

            results = tree.logger.log.values()
            sorted_results= sorted(results, key=lambda x: x.reward_stats["score"], reverse=True)
            stats={'tree_size': len(tree.node_dict),'evaluator':evaluator.__class__.__name__,'time': delta_time}
            stats_obj=NetGAPStats(node.state,sorted_results[0],stats)
            history.append(stats_obj)
        
    return history

def app_star(args):
    return app(*args)

def parallel_imap(items):
    results=[]
    
    with Pool(6) as p:
        try:
            for result in tqdm(p.imap(app_star,items),total=len(items)):
                results.extend(result)
            p.close()
            return results
        except KeyboardInterrupt:
            print("ending everyhing")
            pool.terminate()
            return None

if __name__ == '__main__':
    
    
    num_samples=1
    num_batches=1
   
    time_limit=None
    iteration_limit=None
    evaluator='b'
    
    print(str(sys.argv))
    argv=sys.argv
    opts, args = getopt.getopt(argv[1:],"s:b:t:i:")
    for opt, arg in opts:
        
        if opt == '-s':
            num_samples = int(arg)
            print ('samples',num_samples)
        if opt == "-b" :
            num_batches = int(arg)
            print ('batches',num_batches)
         
        if opt == "-i":
            iteration_limit = int(arg)
            print ('it',iteration_limit)
        elif opt =="-t":
            time_limit=float(arg)
            print ('time',time_limit)
    
    samples_per_thread=math.ceil(num_samples/num_batches)
    print("samples per batch" ,  samples_per_thread)
    
    jobs = [(samples_per_thread,iteration_limit,time_limit,i) for i in range(num_batches)]
    history=parallel_imap(jobs)
    print(len(history))
    print(history[0])
    
    filename='comparison' 
    
    if time_limit is not None:
        filename+="_time"+str(int(time_limit))
    elif iteration_limit is not None:
        filename+="_its"+str(int(iteration_limit))
    
    now = datetime.datetime.now()
    date= f"{now.month}{now.day}{now.hour}{now.minute}"
    filename += f"_{date}"
    
    saver = ComparisonSaver('C:/Users\smora\Documents\LiU\Workspace\\NetGAP\data\comparisons')
    saver.save(history,filename)
    
    sys.exit()

