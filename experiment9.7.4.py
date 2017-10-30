"""
Proof-of-concept experiment. Can I evolve a model that achieves successful 
switching when DURING THE TRAINING the basal ganglia (BG) are connected to 
the hidden layer of the sensorimotor (SM) SNN? 

My idea:
The BG neurons will go through desynchronization as follows:
    * Fire at the same time every X ticks (X is defined by chosen frequency
      from beta range [12-30Hz]). This is SYNC regime.
    * 50 ticks before the SM SNN has to switch, start firing randomly. This is 
      DESYNC regime (that theoretically helps in the human brain to switch
      btw motor states)
    * 50 ticks after the SM SNN has switched, the BG neurons again start firing
      in SYNC
A successfully train model will allow testing several things:
    * Is the SM SNN switching when the BG does not DESYNC, as it did during the
      training?
    * Will the CTX be able to switch the SNN outside the pre-trained DESYNC time 
      interval

Josh's idea:
    * Evolve the SNN with the CTX and the BG, all of the connection are to be 
      evolved.
    * Change the fitness function such that each network is simulated twice:
        - With BG always in SYNC, the target - no switching of the outputs, 
          regardless of the command signal from CTX
        - With BG always in DESYNC, the target - switching happens normally, 
          when CTX provides the control signal  

First, I will run the Josh's setup. It'll be exp 9.7.3      
"""

import numpy as np
from ga import GeneticAlgorithm
from archive_results import archive_results

# Name of the experiment:
exp_id = 'ROMAN_9.7.04_TEST02_BGrand_sparse_fixwts'    
# Set the SNN params:
node_types                  = ['tonic_spike']
n_nodes_input               = 1
n_nodes_cortex              = 2
n_nodes_hidden              = 10
n_nodes_output              = 1
intralayer_connections_flag = False
add_bias                    = False
out_ma_len                  = 20   # The length of the window for the moving average that averages firing of the hidden neurons for the update of the output neurons
tau                         = 50.0  # time constant for all output neurons (used to update their state)
max_ticks                   = 1000
switch_tick                 = 500
# Set the BG params:
bg_nodes                    = 10
ERD_tick                    = switch_tick-50
ERS_tick                    = switch_tick+50  
bg_flag                     = False
bg_sync_freq                = 25

# Set the genetic algorithm params:
if intralayer_connections_flag:    
    len_indv = (n_nodes_cortex * n_nodes_hidden + 
                n_nodes_input * n_nodes_hidden +
                bg_nodes * n_nodes_hidden +
                n_nodes_hidden * n_nodes_hidden
                + n_nodes_hidden * n_nodes_output)
else:
    len_indv = (n_nodes_cortex * n_nodes_hidden + 
                n_nodes_input * n_nodes_hidden + 
                bg_nodes * n_nodes_hidden +
                n_nodes_hidden * n_nodes_output)
if add_bias:
    len_indv += n_nodes_hidden
# DEBUG:
print "Individual has length =", len_indv    
pop_size          = 150
num_par           = 50
prob_mut          = 0.05
prob_xover        = 0.9
prob_survival     = 0.05
prob_weight_reset = 0.001
max_gen           = 300
weight_range      = 50.0   
mut_range         = 5.0
target_values     = np.zeros((1,2))
target_values[0,0] = 5.0
target_values[0,1] = 15.0
#target_values     = np.array([5.0, 10.0])

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
                      # SNN stuff:
                      node_types=node_types,
                      n_nodes_input=n_nodes_input,
                      n_nodes_cortex=n_nodes_cortex,
                      n_nodes_hidden=n_nodes_hidden,
                      n_nodes_output=n_nodes_output,
                      intralayer_connections_flag=intralayer_connections_flag,
                      add_bias=add_bias,
                      out_ma_len=out_ma_len,
                      tau=tau,
                      max_ticks=max_ticks,
                      switch_tick=switch_tick,
                      # BG stuff:
                      bg_nodes=bg_nodes,
                      ERD_tick=ERD_tick,
                      ERS_tick=ERS_tick,
                      bg_flag=bg_flag,
                      bg_sync_freq=bg_sync_freq)

ga.evolve()

archive_results(exp_id, ga)   


