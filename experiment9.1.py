"""
Experiment to test if different values can be achieved on output nodes. 
Previously, it was established that two identical values can be learned.
"""

# from snn import SpikingNeuralNetwork
from ga import GeneticAlgorithm
# from fitness import FitnessFunction
import os
import shutil

# Name of the experiment:
exp_id = '2OUT_diff_targ_tonic_spike_fit2min_tau50_ma30'    
# Set the SNN params:
node_types           = ['tonic_spike']
n_nodes_input        = 2
n_nodes_hidden       = 20
n_nodes_output       = 2
add_bias             = True
out_ma_len           = 30   # The length of the window for the moving average that averages firing of the hidden neurons for the update of the output neurons
tau                  = 50.0  # time constant for all output neurons (used to update their state)
weight_epsilon       = 0.01
max_ticks            = 1000  

# Set the genetic algorithm params:
len_indv = n_nodes_input * n_nodes_hidden + n_nodes_hidden * n_nodes_hidden + n_nodes_hidden * n_nodes_output
if add_bias:
    len_indv += n_nodes_hidden
# DEBUG:
print "Individual has length =", len_indv    
pop_size          = 50
num_par           = 20
prob_mut          = 0.05
prob_xover        = 0.9
prob_survival     = 0.05
prob_weight_reset = 0.001
max_gen           = 500
weight_range      = 50.0   
mut_range         = 5.0
target_values     = [10.0, 20.0]

ga = GeneticAlgorithm(len_indv=len_indv, 
                      pop_size=pop_size, 
                      num_par=num_par, 
                      prob_mut=prob_mut, 
                      prob_xover=prob_xover,
                      prob_survival=prob_survival,
                      prob_weight_reset=prob_weight_reset,
                      max_gen=max_gen,
                      weight_range=weight_range,
                      mut_range=mut_range,
                      target_values=target_values,
                      node_types=node_types,
                      n_nodes_input=n_nodes_input,
                      n_nodes_hidden=n_nodes_hidden,
                      n_nodes_output=n_nodes_output,
                      add_bias=add_bias,
                      out_ma_len=out_ma_len,
                      tau=tau,
                      weight_epsilon=weight_epsilon,
                      max_ticks=max_ticks)

ga.evolve()

# Housekeeping to save all pertinent experiment results:
script_path = os.getcwd()   
exp_path = script_path + '\\' + exp_id
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
# Pickle the champions for future analysis:
import pickle    
pickle.dump( ga.champions, open( "champions.p", "wb" ) )
pickle.dump( ga, open( "ga.p", "wb") )
# to load pickle (need to go to the experiment folder):
# champions = pickle.load( open( "champions.p", "rb" ) )    
### Throw all of the intermidiate files into the experiment folder
source = os.listdir(script_path)
for files in source:
    if files.endswith(".txt") | files.endswith(".png") | files.endswith(".p"):
        shutil.move(files,exp_path)
    if files.endswith(".py"):
        shutil.copy(files,exp_path)
                

# Much easier saving to .txt function:
# np.savetxt('node_coors.txt', substrate.nodes, fmt = '%3.3f ')    


