"""
Class that evaluates a SNN on a specific task and returns fitness score
An object of this class should have the following functionality:
    * receive a vector of integers (both positive and negative) from a GeneticAlgorithm object
    * convert this vector into weight matrices (needs therefore to know SNN topology!)
    * create a SpikingNeuralNet object and evaluate its performance using "run" method from that class
    * return back the fitness score and/or other metrics
    New in 9.4:
        * The network is evolved to learn to switch from state 1 to state 2 and 
          then back to state 1. There are two target outputs but two switch 
          ticks. 
"""

import numpy as np
np.set_printoptions(precision=3)
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
            """
            # DEBUG:
            # print "network.out_states_history=",network.out_states_history[:,idx]
            for tick in xrange(0,network.max_ticks):
                print "=== Tick",tick,"==="
                print "output_node[",idx,"] =", network.out_states_history[tick,idx]

            for tick in xrange(0,network.max_ticks):
                print "=== Tick",tick,"==="
                print "output_node[",idx,"] - target_vals[",idx,"]=", network.out_states_history[tick,idx] - self.target_vals[idx]

            for tick in xrange(0,network.max_ticks):
                print "=== Tick",tick,"==="
                print "SQRT(output_node[",idx,"] - target_vals[",idx,"])**2=", np.sqrt( (network.out_states_history[tick,idx] - self.target_vals[idx])**2 )                
            """    
            rmse_this = np.sqrt(((network.out_states_history[100:network.switch_tick[0],idx] - self.target_vals[idx, 0]) ** 2).mean())
            rmse_this += np.sqrt(((network.out_states_history[network.switch_tick[0]+100:network.switch_tick[1],idx] - self.target_vals[idx, 1]) ** 2).mean())
            rmse_this += np.sqrt(((network.out_states_history[network.switch_tick[1]+100:,idx] - self.target_vals[idx, 0]) ** 2).mean())
            """
            # DEBUG:
            print "RMSE_this =", rmse_this
            """
            rmse_total += rmse_this
            rmse_this = 0.0
        
        rmse_total = rmse_total / network.n_nodes_output
        
        return (1.0 / (1.0 + rmse_total))
    
