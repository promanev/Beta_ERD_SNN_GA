import numpy as np
np.set_printoptions(precision=3)
import math as m
import random as rd
import pylab as plb
# from bokeh.plotting import figure, output_file, show

from fitness import FitnessFunction
from snn import SpikingNeuralNetwork

def mean(numbers): # Arithmetic mean fcn
    return float(sum(numbers)) / max(len(numbers), 1)
# end MEAN fcn

class GeneticAlgorithm(object):
    # Comments:
    #
    # 
    def __init__(self, 
                 len_indv=40, 
                 pop_size=100, 
                 num_par=40, 
                 prob_mut=0.05, 
                 prob_xover=0.9,
                 prob_survival=0.05,
                 prob_weight_reset=0.01,
                 max_gen=1000,
                 weight_range=50.0,
                 mut_range=10.0,
                 target_values=None,
                 champions=None,
                 # SNN params:
                 node_types=['tonic_spike'],
                 n_nodes_input=None,
                 n_nodes_cortex=None,
                 n_nodes_hidden=None,
                 n_nodes_output=None,
                 intralayer_connections_flag=False,
                 add_bias=True,
                 out_ma_len=20,
                 tau=5.0,
                 weight_epsilon=0.01,
                 max_ticks=1000,
                 switch_tick=500):
        # GA params:
        # len_indv - length of vector representing a single individual
        # pop_size - number of individuals in the population
        # num_par - number of parents selected to reproduce
        # prob_mut - probability of mutation
        # prob_xover - probability of crossover
        # prob_survival - a very low probability that a low-fitness pop.member is not replaced and stays in the population (improves diversity)
        # prob_weight_reset - a very small probability that instead of slightly shifting a weight, it is picked randomly anew (helps to get out of local optima)
        # max_gen - maximum generations alotted for evolution
        # weight_range - weights are generated within [-weight_range, weight_range]
        # mut_range - when mutation occurs, a weight is changed by a random integer from [-mut_range, mut_range]
        # target_values - the values that output neurons should match
        # champions - best 10 individuals to be pickled
        
        # SNN params:
        # node_types - string name(s) of types of spiking neurons to be used for hidden layer
        # n_nodes_input - number of input nodes (not including the bias neuron)
        # n_nodes_cortex - number of neurons that represent the desire to output one set of output values or another
        # n_nodes_hidden - number of hidden nodes (these are the only spiking neurons)
        # n_nodes_output - number of output nodes (CTRNN-type nodes)
        # intralayer_connections_flag - use True if you want to wire hidden neurons with each other
        # add_bias - True/False flag that adds/removes bias node from input layer that always has 1.0 state
        # out_ma_len - length of the moving average window that is used to approximate firing of hidden neurons
        # tau - time constant for output neurons (larger values make for more slowly-changing neurons)
        # weight_epsilon - weights below this value are considered equal to 0
        # max_ticks - the number of time steps used to simulate SNN activity
        # switch_tick - the tick when the cortical neurons will switch from one type of firing to another
        
        # GA stuff:
        self.len_indv=len_indv
        self.pop_size=pop_size
        self.num_par=num_par
        self.prob_mut=prob_mut
        self.prob_xover=prob_xover
        self.prob_survival=prob_survival
        self.prob_weight_reset=prob_weight_reset
        self.max_gen=max_gen
        self.weight_range=weight_range
        self.mut_range=mut_range
        self.target_values=target_values
        self.champions=champions
        
        # SNN stuff:
        self.node_types=node_types
        self.n_nodes_input=n_nodes_input
        self.n_nodes_cortex=n_nodes_cortex
        self.n_nodes_hidden=n_nodes_hidden
        self.n_nodes_output=n_nodes_output
        self.intralayer_connections_flag=intralayer_connections_flag
        self.add_bias=add_bias
        self.out_ma_len=out_ma_len
        self.tau=tau
        self.weight_epsilon=weight_epsilon
        self.max_ticks=max_ticks
        self.switch_tick=switch_tick
    # end INIT   
    
    def save_weights_txt(self, weight_matrix, filename):
        """
        Method takes in weight matrix and saves ii in a .txt file row by row
        weight_matrix - data to be stored
        filename - name of the file containing the weight martix
        """
        fid = open(filename,'w+')
        num_rows = weight_matrix.shape[0]
        num_columns = weight_matrix.shape[1]
       
        for row in xrange(0, num_rows):
            for idx in xrange(0, num_columns):
                fid.write(str(weight_matrix[row,idx])+' ')
            fid.write("\n")    
        # end FOR
        fid.close() 
    
    def convert_pop_member(self, pop_member, save_flag=False, fname='weights'):
        """
        Converts a 1-D vector into three weight matrices
        pop_member - 1D vector containing all of the weights for all of the 
                     weight matrices
        save_flag - (optional) boolean flag to decide whether to save weight 
                    matrices to .txt files
        fname     - a string template to be used for naming .txt files. Convention
                    is '%Generation_number%_%best_number%' as there is at least 
                    10 best saved every time. '.txt' is not necessary as it is 
                    added inside this method
        """
        n_nodes_input = self.n_nodes_input
        if self.add_bias:
            n_nodes_input+=1
        if self.n_nodes_cortex is not None:
            n_nodes_cortex = self.n_nodes_cortex
        n_nodes_hidden = self.n_nodes_hidden
        n_nodes_output = self.n_nodes_output
        
        w_ih = np.zeros((n_nodes_input, n_nodes_hidden), dtype=float)
        if self.intralayer_connections_flag:
            w_hh = np.zeros((n_nodes_hidden, n_nodes_hidden), dtype=float)
        else:
            w_hh = None
            
        if self.n_nodes_cortex is not None:
            w_ch = np.zeros((n_nodes_cortex, n_nodes_hidden), dtype=float)
        else:
            w_ch = None
            
        w_ho = np.zeros((n_nodes_hidden, n_nodes_output), dtype=float)
        
        pointer = 0
        
        for row in xrange(0, n_nodes_input):
            start_id = row * n_nodes_hidden
            end_id = start_id + n_nodes_hidden
            
            w_ih[row,] = pop_member[start_id:end_id]
        
        pointer = end_id
        
        if self.n_nodes_cortex is not None:
            for row in xrange(0, n_nodes_cortex):
                start_id = row * n_nodes_hidden + pointer
                end_id = start_id + n_nodes_hidden
            
                w_ch[row,] = pop_member[start_id:end_id]
        
        pointer = end_id
        
        if self.intralayer_connections_flag:
            for row in xrange(0, n_nodes_hidden):
                start_id = row * n_nodes_hidden + pointer
                end_id = start_id + n_nodes_hidden
                        
                w_hh[row,] = pop_member[start_id:end_id]
        
            pointer = end_id
        
        for row in xrange(0, n_nodes_hidden):
            start_id = row * n_nodes_output + pointer
            end_id = start_id + n_nodes_output            
            
            w_ho[row,] = pop_member[start_id:end_id]
        
        if save_flag:
            filename = fname + '_IH.txt'
            self.save_weights_txt(w_ih, filename)
            
            if self.intralayer_connections_flag:
                filename = fname + '_HH.txt'
                self.save_weights_txt(w_hh, filename)   
            
            if self.n_nodes_cortex is not None:
                filename = fname + '_CH.txt'
                self.save_weights_txt(w_ch, filename)                   
            
            filename = fname + '_HO.txt'
            self.save_weights_txt(w_ho, filename)            
            
        return (w_ih, w_ch, w_hh, w_ho)        
        
    def get_fitness(self, pop_member):
        """
        This method depends on FitnessFunction class. It should take a single pop.member and make a 
        SpikingNeuralNetwork object based on the vector (becomes a series of weight matrices) and
        SNN parameters that are provided to GeneticAlgorithm object. Then, this SNN is sent 
        to "evaluate" method of FitnessFunction class that returns the fitness score. 
        """
        
        # Init a SNN object
        network = SpikingNeuralNetwork(node_types=self.node_types,
                                       n_nodes_input=self.n_nodes_input,
                                       n_nodes_hidden=self.n_nodes_hidden,
                                       n_nodes_output=self.n_nodes_output,
                                       add_bias=self.add_bias,
                                       out_ma_len=self.out_ma_len,
                                       tau=self.tau,
                                       weight_epsilon=self.weight_epsilon,
                                       max_ticks=self.max_ticks)
        
        # Convert the vector into weight matrices:
        (w_ih, w_ch, w_hh, w_ho) = self.convert_pop_member(pop_member, save_flag=False)
        
        # update the SNN with weight matrices:
        network.from_matrix(w_ih, w_hh, w_ho, w_ch, self.node_types)    
            
        # Init a FitnessFunction object to evaluate the SNN:
        fitness_fcn = FitnessFunction(self.target_values)
        
        # Now evaluate the network:
        fitness_score = fitness_fcn.evaluate(network)    
        
        return fitness_score
    # end getFitness
    
    def select_parents(self, fitness): 
        # function for selecting parents using stochastic universal sampling + identifying worst pop members to be replaced
        # calculate normalized fitness:
        total_fitness = m.fsum(fitness)
        norm_fitness = np.zeros((len(fitness)))
        for i in xrange(0,len(fitness)):
            norm_fitness[i] = fitness[i] / total_fitness
        # end FOR

        # create cumulative sum array for stochastic universal sampling (http://www.geatbx.com/docu/algindex-02.html#P472_24607):
        cumul_fitness = [[0 for x in range(len(fitness))] for y in range(2) ] # one row for cumul fitness values, the other for population member IDs
        count = 0
        norm_fitness_temp = norm_fitness # make a copy of normalized fitness to extract indices
        while count < len(fitness):
            # find max fit and its ID:
            max_fit = max(norm_fitness)
            max_fit_id = np.argwhere(norm_fitness_temp==max_fit)

            # store cumulative norm fitness (add the sum of all previous elements)
            if count==0:
                cumul_fitness[0][count]=max_fit
                cumul_fitness[1][count]=int(max_fit_id[0])
            else:
                cumul_fitness[0][count]=max_fit+cumul_fitness[0][count-1] # have to add previous fitnesses
                cumul_fitness[1][count]=int(max_fit_id[0]) # record the ID of the pop member
        
            # remove this min fit from temp fit array:
            norm_fitness_new = np.delete(norm_fitness, norm_fitness.argmax())
            norm_fitness = norm_fitness_new
            count = count + 1
        # end WHILE
        
        # roll a roulette wheel once to select 10 parents:    
        # generate the "roll" btw 0 and 1/num_parents:
        random_pointer = rd.uniform(0, 1.0/self.num_par)
        # keep parent IDs here:
        par_ids = []
        count = 0 # counter to go through potential parents. Keep this counter here, so that next parent is searched starting where the last one was found, not from the beginning
        par_count = 0.0 # counter for threshold adjustment
    
        # detect where roulette wheel prongs ended up:
        for x in xrange(0, self.num_par):
            found_flag = 0     
            while found_flag==0:
                if cumul_fitness[0][count]>(random_pointer + par_count / self.num_par): # for each successive parent the roulette wheel pointer shifts by 1/num_par      
                    par_ids.append(cumul_fitness[1][count])
                    found_flag = 1
                else:
                    count = count + 1
                # end IF
            # end WHILE
            par_count = par_count + 1
        # end FOR

        # IDs of pop.members to be replaced:
        last_inx = len(cumul_fitness[1])
        worst_ids = cumul_fitness[1][last_inx - self.num_par:last_inx]
        return (par_ids, worst_ids) 
    # end SELECTPARENTS
    
    def produce_offspring(self, pop, par_ids, worst_ids, was_changed):
        # produces children using uniform crossover and bitflip mutation
        num_pairs = len(par_ids)/2
        for x in xrange(0,num_pairs):
            id1 = rd.choice(par_ids)
            del par_ids[par_ids.index(id1)] # remove the id from the list so it's not chosen twice
            id2 = rd.choice(par_ids)
            del par_ids[par_ids.index(id2)]
            # generate mask for crossover:
            mask=np.random.randint(2, size=(len(pop[id1]),))
            # generate a pair of children bitwise:
            child1=[]
            child2=[]
            if rd.uniform(0.0,1.0) < self.prob_xover:
                for y in xrange(0,len(pop[id1])):
                    if mask[y]==0:
                        child1.append(pop[id1][y])
                        child2.append(pop[id2][y])
                    else:
                        child1.append(pop[id2][y])
                        child2.append(pop[id1][y])
                    # end IF
                # end FOR
            else:
                child1,child2=pop[id1],pop[id2] # if unsuccessful, children are exact copies of parents
            # end IF            
        
            # apply mutation:
            # UPDATE: since the weights are no longer binary but rather integers, the previous code was 
            # re-written. Now, if the weight is to be mutated, it has a random integer from 
            # [-mutation_range; mutation_range] added to it. This ensures that mutations are not too drastic.
            for y in xrange(0, self.len_indv):
                # roll a die for the first child:
                if rd.uniform(0.0,1.0) < self.prob_mut:
                    child1[y]+=np.random.randint(-self.mut_range-1,self.mut_range+1)
                    # Check that result is within bounds:
                    if child1[y]>self.weight_range:
                        child1[y]=self.weight_range
                    if child1[y]<-self.weight_range:
                        child1[y]=-self.weight_range                        
                # end IF
            
                # roll a die for the second child:
                if rd.uniform(0.0,1.0) < self.prob_mut:
                    child2[y]+=np.random.randint(-self.mut_range-1,self.mut_range+1)
                    if child2[y]>self.weight_range:
                        child2[y]=self.weight_range
                    if child2[y]<-self.weight_range:
                        child2[y]=-self.weight_range                                            
                # end IF 

                # There is also a very low probability that a weight would be assigned a completely new 
                # random integer rather than changed slightly, as above:
                if rd.uniform(0.0,1.0) < self.prob_weight_reset:
                    child1[y]=np.random.randint(-self.weight_range-1,self.weight_range+1)

                if rd.uniform(0.0,1.0) < self.prob_weight_reset:
                    child2[y]=np.random.randint(-self.weight_range-1,self.weight_range+1)                    
            # end FOR            

            # replace individuals with lowest fitness but leave a small chance (5%) that the worst member will survive:
            if rd.uniform(0.0,1.0) < 1-self.prob_survival:
                pop[worst_ids[x]]=child1 # Ex: for 5 pairs, replace 0-th and 5-th; 1-st and 6-th ...
                was_changed[worst_ids[x]]=0 # mark that the pop.member was changed and its fitness should be re-evaluated
            if rd.uniform(0.0,1.0)< 1-self.prob_survival:
                pop[worst_ids[x+num_pairs]]=child2        
                was_changed[worst_ids[x+num_pairs]]=0 # 0 - need to re-evaluate fitness 
            # end IF
        # end FOR
        return (pop,was_changed)
    # end PRODUCEOFFSPRING

    def evolve(self, verbose=True, save_results=True):
        # to keep track of performance:
        best_fit=np.zeros((self.max_gen))
        avg_fit=np.zeros((self.max_gen))
        # fitness array:
        fitness = np.zeros((self.pop_size))
        # array to keep track of unchanged population members (to avoid calculating their fitness):
        was_changed = np.zeros((self.pop_size)) # 0 - means the member was changed and needs to be re-assessed, 1 - no need to re-assess its fitness

        # generate initial population:
        # Both positive and negative weights are allowed and should be within [-weight_range, weight_range]    
        pop = [np.random.randint(-self.weight_range-1,self.weight_range+1, size=(self.len_indv,)) for i in xrange(0, self.pop_size)]

        # main cycle:
        for gen in xrange(0,self.max_gen):
            if verbose:
                print "Gen #",gen
            
            # calculate fitness for new population members: 
            for member in xrange(0, self.pop_size):
                if was_changed[member]==0:
                    fitness[member]=self.get_fitness(pop[member])
                    was_changed[member]=1
                # end IF
            # end FOR
            
            # store this generation's best and avg fitness:
            best_fit[gen]=max(fitness)
            avg_fit[gen]=mean(fitness)
            if verbose:
                print "Best fit=",best_fit[gen],"Avg.fit=",avg_fit[gen]
    
            # select parents for mating:
            (par_ids, worst_ids) = self.select_parents(fitness)
    
            # replace worst with children produced by crossover and mutation:
            (pop,was_changed) = self.produce_offspring(pop, par_ids, worst_ids, was_changed)
            
    
        # end FOR === MAIN CYCLE ===
        if save_results:
            # SAVE 5 BEST:
            self.champions = np.zeros((5, self.len_indv), dtype='float')    
            best5_ids = fitness.argsort()[-5:][::-1]
            for idx in xrange(0,len(best5_ids)):
                best_name = 'best_weights'+str(0)+str(idx+1)
                self.convert_pop_member(pop[best5_ids[idx]], save_flag=True, fname=best_name)
                self.champions[idx,] = pop[best5_ids[idx]]
            # end FOR

            # plot best behavior:
            best_network = SpikingNeuralNetwork(node_types=self.node_types,
                                                n_nodes_input=self.n_nodes_input,
                                                n_nodes_hidden=self.n_nodes_hidden,
                                                n_nodes_output=self.n_nodes_output,
                                                add_bias=self.add_bias,
                                                out_ma_len=self.out_ma_len,
                                                tau=self.tau,
                                                weight_epsilon=self.weight_epsilon,
                                                max_ticks=self.max_ticks)
        
            # Convert the vector into weight matrices:
            (w_ih, w_ch, w_hh, w_ho) = self.convert_pop_member(pop[best5_ids[0]], save_flag=False)
        
            # update the SNN with weight matrices:
            best_network.from_matrix(w_ih, w_hh, w_ho, w_ch, self.node_types)      
            
            print "Using the switch tick =", best_network.switch_tick
            # now plot behavior: 
            self.show_behavior(best_network, file_tag='switch@500') 
            
            # now test the behavior with the switching ticked moved away from
            # the value used during the training:
            best_network.switch_tick = 700
            print "Using the switch tick =", best_network.switch_tick                
            self.show_behavior(best_network, file_tag='switch@700')
            
        # FITNESS PLOT:    
        #
        # create a new plot:
        gens=np.arange(0,self.max_gen)
        plb.plot(gens,best_fit,'b')
        plb.plot(gens,avg_fit,'r')
        plb.title('Fitness plot')
        plb.xlabel('Generation')
        plb.ylabel('Fitness')
        plb.savefig('fitness_plot.png')
        plb.show()
    # end EVOLVE        

    def show_behavior(self, network, file_tag=None):
        """
        This method plots behavior of output nodes vs. the targets
        """
        
        # network.run(verbose=True)
        network.run()
        ticks=np.arange(0, network.max_ticks)
        # Plot the behavior of the output nodes:    
        for node in xrange(0,network.n_nodes_output):
            plb.figure(figsize=(12.0,10.0))
            plb.plot(ticks, network.out_states_history[:,node],'r',label='Output node '+str(node))
            target_vector_01 = np.full((1, network.switch_tick), self.target_values[node,0])
            target_vector_02 = np.full((1, network.max_ticks - network.switch_tick), self.target_values[node,1])
            target_vector = np.hstack((target_vector_01,target_vector_02))
            # print "Targets' vector shape:", target_vector.shape
            # print "Ticks' shape:", ticks.shape
            plb.plot(ticks, target_vector[0,:], 'b', label='Target for node '+str(node))
            plb.title('Behavior plot')
            plb.xlabel('Simulation tick')
            plb.ylabel('Node output vs. target')
            plb.legend()
            plb.savefig('behavior_plot_out_node_'+str(node)+'_'+file_tag+'.png', dpi=300)
            plb.show()
            
        # Plot the spiking activity of hidden nodes (in 1000 ticks epochs):
        # 
        # First, convert binary spike data into plottable format:
        spike_neuron = []
        spike_tick = []
        for tick in xrange(0, network.max_ticks):
            for node in xrange(0, network.n_nodes_hidden):
                if network.hid_states_history[tick,node]>0.0:
                    spike_neuron.append(node+1)
                    spike_tick.append(tick)            
        
            
            
            
        # DEBUG:
        # print "Outputting the hidden layer spikes:"
        # for tick in xrange(0, self.max_ticks):
        #     print network.hid_states_history[tick,]
            
        # num_epoch = int( np.floor(self.max_ticks-1 / 1000) ) + 1
        
        # DEBUG:
        # print "Total ticks =", self.max_ticks
        # print "Number of 1000 tick epochs (including the last that may be < 1000) =", num_epoch
        
        current_figure = plb.figure(figsize=(12.0,10.0))
        plb.plot(spike_tick, spike_neuron,'k.')
        plb.title('Spike Train Raster Plot')
        plb.xlabel('Time, ticks')
        plb.ylabel('Neuron #')
        axes = plb.gca()
        axes.set_ylim([0,self.n_nodes_hidden+1])
        axes.set_xlim([0,self.max_ticks+1])
        plb.show()
        current_figure.savefig('spikes_plot_'+file_tag+'.png', dpi=300)                

            
    # END SHOW_BEHAVIOR
    
    
if __name__ == 'main':
    #
    #
    print "Class GeneticAlgorithm"
