"""
Class that evaluates a SNN on a specific task and returns fitness score
An object of this class should have the following functionality:
    * receive a vector of integers (both positive and negative) from a GeneticAlgorithm object
    * convert this vector into weight matrices (needs therefore to know SNN topology!)
    * create a SpikingNeuralNet object and evaluate its performance using "run" method from that class
    * return back the fitness score and/or other metrics
"""

import numpy as np
import pylab as plb
# from snn import SpikingNeuralNetwork



class FitnessFunction(object):
    """
    
    """
    def __init__(self, target_vals):
        
        self.target_vals=target_vals 
        
        
    def evaluate(self, network):
        """
        network - the instance of SNN class with all relevant fields filled 
                  out (e.g., weight matrices, etc.)
        """
        network.run()
        rmse_total = 0.0
        rmse_this = 0.0
        for idx in xrange(0, network.n_nodes_output):
            rmse_this = np.sqrt(((network.out_states_history - self.target_vals[idx]) ** 2).mean())
            rmse_total += rmse_this
            rmse_this = 0.0
        
        rmse_total = rmse_total / network.n_nodes_output
        
        return 1000.0 * (1.0 / (1.0 + rmse_total))
    


