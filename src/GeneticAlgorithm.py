import numpy as np
import math

class SelectionOperator:
    def select(self,fitness):
        return np.ones((len(fitness),2))*fitness[0]
    
class CrossoverOperator:
    def cross(self,p1,p2):
        return Individual((len(p1.inclusion),len(p1.allocation)))
    
class MutationOperator:
    def mutate(self,i):
        return Individual((len(i.inclusion),len(i.allocation)))


class RouletteWheelSelector(SelectionOperator):
    
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
        
    def select(self,fitness,n=None):
        if n is None:
            n=len(fitness)
            
        sorted_idxs=np.argsort(fitness)[0:int(n)]
        
        probabilities=(fitness[sorted_idxs])/sum(fitness[sorted_idxs])
        try:
            return self.rng.choice(sorted_idxs,size=(int(len(fitness)/2),2),p=probabilities)
        except Exception as error:
            print("An exception occurred:", error)
            print(fitness[sorted_idxs])
            print(probabilities)
            
class ExponentialRankSelector(SelectionOperator):
    
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
        
    def select(self,fitness,exp=1,n=None):
        if n is None:
            n=len(fitness)
            
        sorted_idxs=np.argsort(fitness)[0:int(n)]
        
        probabilities=(fitness[sorted_idxs]**exp)/sum(fitness[sorted_idxs]**exp)
        try:
            return self.rng.choice(sorted_idxs,size=(int(len(fitness)/2),2),p=probabilities)
        except Exception as error:
            print("An exception occurred:", error)
            print(fitness[sorted_idxs])
            print(probabilities)

class ReverseSequenceMutation(MutationOperator):
    
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
    
    def mutate(self,individual,rate=0.1):
         
        if self.rng.random()<rate:
 
           
            cutoff_1 = self.rng.choice(len(individual.genome)-1)
            cutoff_2 = min(cutoff_1 + max(2,self.rng.choice(math.floor(len(individual.genome)/4))),len(individual.genome)-1)
   
            individual.genome[cutoff_1:cutoff_2]=np.flip(individual.genome[cutoff_1:cutoff_2])
 
    
        return individual
        
    
class PartiallyMappedCrossover(CrossoverOperator):
    
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
    
    def cross(self,p1,p2):
        #print(p1,p2)
        
        cutoff_1, cutoff_2 = np.sort(self.rng.choice(np.arange(len(p1.genome)+1), size=2, replace=False))
        
        def one_offspring(p1, p2):
            offspring=Individual(len(p1.genome))
                
                # Copy the mapping section (middle) from parent1
            offspring.genome[cutoff_1:cutoff_2] = p1.genome[cutoff_1:cutoff_2]

                # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1.genome))]):
                candidate = p2.genome[i]
                while candidate in p1.genome[cutoff_1:cutoff_2]: 
                    candidate = p2.genome[np.where(p1.genome == candidate)[0][0]]
                offspring.genome[i] = candidate
            return offspring

        offspring1 = one_offspring(p1, p2)
        offspring2 = one_offspring(p2, p1)

        return offspring1, offspring2


class ReverseSequenceMutation(MutationOperator):
    
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
    
    def mutate(self,individual,rate=0.1):
         
        if self.rng.random()<rate:
 
           
            cutoff_1 = self.rng.choice(len(individual.genome)-1)
            cutoff_2 = min(cutoff_1 + max(2,self.rng.choice(math.floor(len(individual.genome)/4))),len(individual.genome)-1)
   
            individual.genome[cutoff_1:cutoff_2]=np.flip(individual.genome[cutoff_1:cutoff_2])
 
    
        return individual
        
class SinglePointCrossover(CrossoverOperator):
    def __init__(self,seed=None):
        self.rng=np.random.default_rng(seed)
    
    def cross(self,p1,p2):
        
        cutoff = np.random.randint(low=0,high=len(p1.genome))
        def one_offspring(p1, p2):
            offspring=Individual(len(p1.genome))
            offspring.genome[:cutoff] = p1.genome[:cutoff]
            offspring.genome[cutoff:] = p2.genome[cutoff:]

            return offspring

        offspring1 = one_offspring(p1, p2)
        offspring2 = one_offspring(p2, p1)

        return offspring1, offspring2
    
class SimpleBoundedMutation(MutationOperator):
    
    def __init__(self,max_vector,seed=None):
        self.rng=np.random.default_rng(seed)
        self.max_vector=max_vector
    
    def mutate(self,individual,rate=0.1):
         
        if self.rng.random()<rate:
 
            cutoff_1 = self.rng.choice(len(individual.genome)-1)
            cutoff_2 = min(cutoff_1 + max(2,self.rng.choice(math.floor(len(individual.genome)/4))),len(individual.genome)-1)
            bounds=self.max_vector[cutoff_1:cutoff_2]
            individual.genome[cutoff_1:cutoff_2]=[np.random.randint(low=0,high=x) for x in bounds]
 
        return individual

class Individual:
    def __init__(self,size,init_function=None):
            
            if isinstance(size,int):
                self.genome=np.zeros(size,dtype=int)
            elif isinstance(size,tuple): 
                self.genome = [np.zeros(s) for s in size]
            else:
                Exception("Expected int or tuple, got something else instead")
            
            if callable(init_function):
                self.genome=init_function(self.genome)
    
    def __str__(self):
        return str(self.genome)
    
class Population:
    def __init__(self,pop_size,genome_size):
        self.elements=[]
        self.pop_size=pop_size
        self.genome_size=genome_size
        
    def generate(self, pop_size=None, init_fn=None):
              
        if pop_size is None:
            pop_size=self.pop_size
        
        if callable(init_fn):
            self.elements=[Individual(self.genome_size,init_function=init_fn) for _ in range(pop_size)]
        else:
            raise TypeError("Expecting a callable, received something else")

    def __str__(self):
        return str([re.sub('\n', '', str(e)) for e in self.elements])      
           
    def evolve(self,crossover_operator,mutation_operator,evaluator,selector):
        
        if evaluator is None:
            raise TypeError("Expecting an evaluator, received None")
        else:
            fitness = np.array([evaluator.evaluate(e) for e in self.elements])

        if crossover_operator is not None and selector is not None:
            parents = [[self.elements[p[0]],self.elements[p[1]]] for p in selector.select(fitness)]
            offspring = [individual for offspring_pairs in [crossover_operator.cross(p[0],p[1]) for p in parents] for individual in offspring_pairs]
            #offspring[0:5], offspring[5:] =self.elements[0:5],offspring[0:-5]           
            
        elif selector is None and crossover_operator is not None: 
            raise Exception("Cannot crossover without selection operator")
        else:
            offspring=self.elements
               
           
        if mutation_operator is not None:
            offspring = [mutation_operator.mutate(i) for i in offspring]
        
        self.elements=offspring
        return np.max(fitness)
         
class GeneticAlgorithm:
    
    def __init__(self, genome_size, crossover_operator=None, mutation_operator=None, evaluator=None, selector=None):
        
        self.genome_size = genome_size
        
        self.crossover_operator=crossover_operator
        self.mutation_operator=mutation_operator
        self.evaluator=evaluator
        self.selector=selector
          
    def run(self, pop_size, generations, init_fn,get_hist=False):

        population=Population(pop_size,self.genome_size)
        
 
        population.generate(init_fn=init_fn)
        
        fit_hist=[]   
        #pbar = tqdm(range(generations))
        for _ in range(generations): #range(generations): #pbar:
            max_f=population.evolve(crossover_operator=self.crossover_operator,mutation_operator=self.mutation_operator,evaluator=self.evaluator,selector=self.selector)
            #pbar.set_postfix({'fitness': max_f})
            fit_hist.append(max_f)
            
        #print(fit_hist)
        fitness=[self.evaluator.evaluate(e) for e in population.elements]
        
        sorted_idxs=np.argsort(fitness)
        fitness=[fitness[n] for n in sorted_idxs]
        individuals=[population.elements[n] for n in sorted_idxs]
        
        fitness.reverse()
        individuals.reverse()
        if not get_hist:
            return  individuals, fitness
        else:
            return  individuals, fitness, fit_hist