from __future__ import annotations

import time
import math
import random
#from GraphState import GraphState
from typing import Callable, Optional, Tuple, List
# from _typeshed import SupportsLenAndGetItem
from typing import TYPE_CHECKING
import numpy as np

#from watchdog import Watchdog
import copy

if TYPE_CHECKING:
    from _typeshed import SupportsLenAndGetItem

class Action:
    pass

#TODO: implement support for drawing the tree itself

class State:
    # TODO: implement this type of construction and is instance of State everywhere a state is passed/created from
    #  scratch class State(type)

    # def __instancecheck__(cls, instance):
    #    return cls.__subclasscheck__(type(instance))

    # def __subclasscheck__(cls, subclass):
    #    return (hasattr(subclass, 'get_reward') and
    #            callable(subclass.get_reward) and
    #             hasattr(subclass, 'is_terminal') and
    #             callable(subclass.is_terminal))

    def __int__(self, current_player: int = 1):
        self.current_player: int = current_player

    def get_reward(self) -> float:
        # Only needed for terminal states.
        return 0.0

    def is_terminal(self) -> bool:
        return False

    def take_action(self, a: Action) -> State:
        return State()
        # Returns the state which results from taking action
    def get_possible_actions(self) -> SupportsLenAndGetItem[Action]:
        # All actions possible from this state
        return [Action()]

    def get_current_player(self):
        return self.current_player

class Policies:
    
    def __init__(self, seed=None):
        self.rng=np.random.default_rng(seed)

 
    def RandomPolicy(self, node : MonteCarloTreeNode, max_depth: int = 25) -> State:
       
        #print(max_depth)
        state=copy.deepcopy(node.state)
        last_state=state
        depth=node.depth
               
        #max_sim_depth=15
        #max_depth=min(depth+max_sim_depth, max_depth)
        
        
        while not state.is_terminal() and depth < max_depth:
            last_state=state
            try:
                action_list=state.get_possible_actions()
                random_index=self.rng.integers(low=0,high=len(action_list))
                #random_index=0
                action = action_list[random_index]
            except IndexError:
                print(len(state.get_possible_actions()))
                print(state.is_terminal())
                state.draw()
                raise Exception("Non-terminal state has no possible actions: " + str(state))

            else:
                state=state.take_action(action)
                depth+=1
                
#if state.is_terminal():
            #state=last_state
        
        state.depth=depth

         # print(f"\t\trollout finished after {depth} steps")
        return state
    
    def UCTPolicy(self, node: MonteCarloTreeNode, exploration_value: float) -> MonteCarloTreeNode:
        # COMMENT: why is is this choosing always the best node? because UCT should always choose the highest

        best_value = float("-inf")
        best_nodes = []
        for action, child in node.children.values():
            node_value = child.total_reward / child.num_visits \
                + exploration_value * math.sqrt(math.log(node.num_visits) / child.num_visits)

            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
                    
        if len(best_nodes)==0:
            raise Exception("Is either terminal or has no child?")
            

        random_index=self.rng.integers(low=0,high=len(best_nodes))
        #random_index=0
        return best_nodes[random_index]
        
class MonteCarloTreeNode():
    state: State
    def __init__(self, state: State, parent: Optional[MonteCarloTreeNode] = None,depth=0):
        self.state: State = state
        self.is_terminal: Callable = state.is_terminal
        self.is_fully_expanded: bool = state.is_terminal()
        #self.parent: Optional[MonteCarloTreeNode] = parent
        self.last_descendant=None
        self.parents : Dict[MonteCarloTreeNode] = {id(parent):parent}
        self.num_visits: int = 0
        self.total_reward: float = 0
        self.children: dict[str, Tuple[Action, MonteCarloTreeNode]] = {}
        self.depth=0
        self.is_loop=False
        self.last_descendant=None
        #self.is_dead=False

    def __hash__(self):
       # try:
        return hash(self.state)
        #except:
            #return -1
    def __str__(self) -> str:
        s = ["totalReward: %s" % self.total_reward,
             "numVisits: %d" % self.num_visits,

             "isTerminal: %s" % self.is_terminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

class SimulationLogger():
    def __init__(self):
        self.log={}
    
    def add(self, state: GraphState):
        self.log[hash(state)]=state
            
        
class MCST:
    def __init__(self,
                 time_limit: Optional[float] = None,
                 iteration_limit: Optional[int] = None,
                 exploration_constant: Optional[float] = math.sqrt(2),
                 rollout_policy: Callable[[State,float],State] = Policies.RandomPolicy,
                 selection_policy: Callable[[MonteCarloTreeNode, float], MonteCarloTreeNode] = Policies.UCTPolicy,
                 max_depth: Optional[int] = 100,
                 transposition_check: Optional[bool] = False):
        
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.iteration_limit = iteration_limit
            self.limit_type = 'iterations'
        
        self.max_depth = max_depth
        self.transposition_check = transposition_check
        self.node_dict: dict[int,MonteCarloTreeNode]={}
        self.exploration_constant = exploration_constant
        self.rollout: Callable[[State], float] = rollout_policy
        self.get_best_child: Callable[[MonteCarloTreeNode,float], MonteCarloTreeNode] = selection_policy
        
        self.logger=SimulationLogger()
        
        self.root=None
        self.best_reward=0

    def search(self, initial_node: MonteCarloTreeNode) -> Tuple[Action,MonteCarloTreeNode]:
        """searches from the initial state and returns the current best action"""
        
        total_start=len(self.node_dict.values())
        if self.root is None:
            self.root=initial_node
            
        if self.transposition_check:
            if hash(initial_node) not in self.node_dict:
                 self.node_dict[hash(initial_node)]=initial_node
            else:
                 initial_node = self.node_dict[hash(initial_node)]
        else:
            if id(initial_node) not in self.node_dict:
                self.node_dict[id(initial_node)]=initial_node
              
        iterations=0
        if self.limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000 
            
            while time.time() < time_limit:
                self.execute_round(initial_node)
                iterations+=1
                
        else:  
            for i in range(self.iteration_limit):  
                self.execute_round(initial_node)
                iterations+=1

        best_child = self.get_best_child(initial_node, 0)
        #best_child.state.draw()
        action: Action = (action for action, node in initial_node.children.values() if node is best_child).__next__()
        #print(action[0])
        return action, best_child, iterations

    def execute_round(self,start_node):
        selected_node=self.select_node(start_node)
        rollout_state : State = self.rollout(selected_node, self.max_depth)
        reward : float = rollout_state.get_reward()
        self.logger.add(rollout_state)
        self.backpropagate(selected_node, selected_node, reward)


    def select_node(self, node: MonteCarloTreeNode):
        """While a terminal node is not reached return best children to expand from,
            otherwise expand the node and return one of the new children to roll out from
            It is selection and expansion at the same time returns node to roll out from"""
        root=node
                
       # has_non_loop_children=len([n for _,n in node.children.values() if not n.is_loop])>0
        #if node.is_fully_expanded and not has_non_loop_children:
            # node.is_fully_expanded = True
            #REFLECT is marking whole thing as a full loop the berst policy?
            #node.is_loop=True
            #print("full_loop")
           # return node
        
       # has_non_dead_children=len([n for _,n in node.children.values() if not n.is_dead])>0
        #if node.is_fully_expanded and not has_non_dead_children:
            # node.is_fully_expanded = True
            #print("full_dead")
            #return node
        
        while not node.is_terminal(): #what is a terminal node??
            if node.is_fully_expanded:
                child = self.get_best_child(node, self.exploration_constant)
                node=child
            else:
                node = self.expand(node)
                break
        
       # if len(node.state.get_possible_actions())==0:
          #  node.is_dead=True
            #REFLECT: should I exclude dead branches / terminal branches? How to decide when to stop the search
            
        return node

    def expand(self, node: MonteCarloTreeNode) -> Optional[MonteCarloTreeNode]:
        """For every action from current node create new children from that action
        if all actions added, mark current node as fully expanded and return last created children 
        returns last created children in expansion"""

        actions = node.state.get_possible_actions()
        for action in actions:
            
            # REFLECT is fully expanding node the best policy? Or is adding one at a time better? 
            # We currently add one at a time
            
            #REFLECT is returning the last children from expansions the best policy?

            # TODO: turn actions into a hashable object enclosing productions somehow,
            #  current fix is having action and a hash, but action should be hashable by itself
            
            action_hash = f'{action[0].name}in{action[1]}'
            if action_hash not in node.children:
                new_node = MonteCarloTreeNode(copy.deepcopy(node.state).take_action(action), parent=node,depth=node.depth+1)
                
               
                #if action leads to the same state, mark as loop action 
                if hash(new_node) == hash(node):
                    print("is loop: ", action_hash)
                    state.draw()
                    raise Exception("Grammar creates an unforeseen loop")
                    
                else:
                    if not self.transposition_check:
                        node.children[action_hash] = (action,new_node)
                        self.node_dict[id(new_node)] = new_node
                        
                    else:
                        
                        if hash(new_node) not in self.node_dict:
                            self.node_dict[hash(new_node)] = new_node
                            node.children[action_hash] = (action,new_node)
                            
                        elif hash(new_node) in self.node_dict:
                            retrieved_node = self.node_dict[hash(new_node)]
                            node.children[action_hash] = (action, retrieved_node)
                            retrieved_node.parents[id(node)]=node   
                            retrieved_node.depth=min(new_node.depth,retrieved_node.depth)
                            
                            #new_node=self.select_node(retrieved_node)

                            
                        else:
                             raise Exception("Should never reach here")
                
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                
                return new_node

        raise Exception("Should never reach here")
    
    def backpropagate(self,origin,target,reward):
        if target is not None:
            if id(origin) != target.last_descendant:
                target.last_descendant=id(origin)
                target.total_reward+=reward
                target.num_visits += 1
                for p in target.parents.values():
                    self.backpropagate(origin,p,reward)
    
    def print_node(self,node,height):
        for k,(a,n) in node.children.items():
            print("| "*height,k,n.is_fully_expanded, n.num_visits)
            self.print_node(n,height+1)  
    
    def print_tree(self,node):
        print("root")
        height=0
        self.print_node(node,height+1)          