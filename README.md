# NetGAP
NetGAP/NeuralGAP Graph-Grammar based network generator.

## Usage
Example usage of the nework generator can be seen in the jupyter notebook files or in src/main.py

To generate a network one needs to specify a grammar, a use-case and an evaluator. Examples of evaluators are provided in the src/ directory and of grammars are provided in grammar/. The use case is loaded by the evaluator, examples are in data/.

## Grammars
A description of how to create grammars can be found on the original paper: https://arxiv.org/abs/2306.07778

## Evaluators
Several types of evaluators are provided. If starting from scratch, I think the most advanced is the GeneticEvaluatorOptimRouting. It is slow but yields good results. Evaluators usually read the use-case files for evaluating solutions, so that is where I suggest the interface with the use-case to happen.

# General tips
One needs to specify/limit the number of steps/iterations to be taken while looking for a solutions. If one does not know this number but still wants to allow for infinite grammars, I suggest adding final symbols to grammars. I.e. a rule that when taken ends the search.

I suggest starting the search with the most complete graph as possible. (E.g if the desired number of nodes of type x is known to be n, start the search with a graph containing n unconnected nodes of type x)