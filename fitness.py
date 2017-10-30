"""
Class that evaluates a SNN on a specific task and returns fitness score
An object of this class should have the following functionality:
    * receive a set of tagets from the experiment file (stored in the 
      GeneticAlgorithm object as well)
    * receive an actual SNN to be tested (it should already have the weight 
      matrices that were derived from a GA population member)
    * NEW (9.7.03): an SNN is evaluated twice:
        - with the BG in SYNC (outputs should NOT change after receiving a 
          command from CTX)
        - with the BG in DESYNC (outputs should change after the command from
          CTX)
    * Calculate the RMSe for each neuron, for each tick in both simulations
    * Calculate fitness scores as 1/(1 + RMSe)
    * Output the minimal of two as the final fitness score
"""

import numpy as np

class FitnessFunction(object):
    """
    
    """
    def __init__(self, target_vals):
        
        self.target_vals=target_vals 
        
        
    def evaluate(self, network, verbose=False):
        """
        network - the instance of SNN class with all relevant fields filled 
                  out (e.g., weight matrices, etc.)
        """
        # Grace period is the time for which the output node is not required to
        # meet the desired target value. This is done due to their inertia - it 
        # takes time to change the state of the output neuron. The magnitude of 
        # this inertia is defined by the time constant "tau" and is, therefore,
        # chosen to be the grace period value:
        grace_period = int(network.tau)
            
        network.run()
        rmse_total = 0.0
        rmse_this = 0.0
        
        for idx in xrange(0, network.n_nodes_output):
            rmse_this = np.sqrt(((network.out_states_history[grace_period:network.switch_tick,idx] - self.target_vals[idx, 0]) ** 2).mean())
            rmse_this += np.sqrt(((network.out_states_history[network.switch_tick+grace_period:,idx] - self.target_vals[idx, 1]) ** 2).mean())
            rmse_total += rmse_this
            rmse_this = 0.0
        
        rmse_total = rmse_total / network.n_nodes_output
        fitness = 1.0/(1.0 + rmse_total)
        
        return fitness
    
 
    


