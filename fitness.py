"""
Class that evaluates a SNN on a specific task and returns fitness score
An object of this class should have the following functionality:
    * receive a vector of integers (both positive and negative) from a GeneticAlgorithm object
    * convert this vector into weight matrices (needs therefore to know SNN topology!)
    * create a SpikingNeuralNet object and evaluate its performance using "run" method from that class
    * return back the fitness score and/or other metrics
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
            rmse_this = np.sqrt(((network.out_states_history[200:network.switch_tick,idx] - self.target_vals[idx, 0]) ** 2).mean())
            rmse_this += np.sqrt(((network.out_states_history[network.switch_tick+200:,idx] - self.target_vals[idx, 1]) ** 2).mean())
            """
            # DEBUG:
            print "RMSE_this =", rmse_this
            """
            rmse_total += rmse_this
            rmse_this = 0.0
        
        rmse_total = rmse_total / network.n_nodes_output
        
        return (1.0 / (1.0 + rmse_total))
    

class FitnessFunction2(object):
    """
    This fitness function is designed in hopes to get two output nodes to exhibit
    two different states. The previous fitness function leads to two nodes exhibiting 
    states that are in between each individual's targets.
    There are two ways to implement this fitness function that tries to encompass
    two objectives: a) reach your target, b) stay away from other neurons' targets. 
    Namely, two fitness scores from a) and b) can be multiplied (less punishment
    for not reaching one of the objectives) or overall fitness can be the lower of two
    (more punishing, have to be very good at both!)
    """
    def __init__(self, target_vals):
        
        self.target_vals=target_vals 
        
        
    def evaluate(self, network):
        """
        network - the instance of SNN class with all relevant fields filled 
                  out (e.g., weight matrices, etc.)
        """
        network.run()
        # These are regular RMS errors that grow as the SNN behavior deviates from 
        # the desired behavior
        rmse_total = 0.0
        rmse_this = 0.0
        
        # These will calculate how close the output node behavior comes close 
        # to other node's target and penalize based on this:
        proximity_error_this = 0.0
        proximity_error_total = 0.0
        
        output_nodes_indices = np.arange(0, network.n_nodes_output)
        
        for idx in xrange(0, network.n_nodes_output):
            rmse_this = np.sqrt(((network.out_states_history[200:,idx] - self.target_vals[idx]) ** 2).mean())
            rmse_total += rmse_this
            rmse_this = 0.0

            other_output_node_indices = np.delete(output_nodes_indices,idx)
            for other_idx in other_output_node_indices:
                proximity_error_this = np.sqrt(((network.out_states_history[200:,idx] - self.target_vals[other_idx]) ** 2).mean())
                proximity_error_total += proximity_error_this
                proximity_error_this = 0.0
            # Average by the number of comparisons made for a single output neuron:    
            proximity_error_total = proximity_error_total / (network.n_nodes_output - 1)    
        
        # Average by the number of output neurons:
        rmse_total = rmse_total / network.n_nodes_output
        proximity_error_total = proximity_error_total / network.n_nodes_output    
        
        fit_rmse = 1.0 * (1.0 / (1.0 + rmse_total))
        fit_proxy = 1.0 * (1.0 / (1.0 + (1.0 / proximity_error_total) ) )
        return np.fmin(fit_rmse, fit_proxy)    
    


