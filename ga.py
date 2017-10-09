import numpy as np
import math as m
import os
import sys
# VACC only: 
# import subprocess as subp
import random as rd
import time as t
import shutil, errno
import pylab as plb
# from bokeh.plotting import figure, output_file, show

from fitness import FitnessFunction
from snn import SpikingNeuralNetwork

def mean(numbers): # Arithmetic mean fcn
    return float(sum(numbers)) / max(len(numbers), 1)
# end MEAN fcn

"""
def read_text_file(fileName):
    loadFlag = False
    loadAttempt = 0
    attemptMax = 10
    dataVar = []
    while (loadAttempt<attemptMax) & (loadFlag==False):
        if(os.path.exists(fileName)==False):
            # pause for 0.5 second:
            t.sleep(0.5)
            loadAttempt = loadAttempt + 1
            print("Load attempt "+str(loadAttempt))
        else:
            fid = open(fileName,'r')
            # read just one line from fit.txt
            for line in fid:
                dataVar += [float(line)]
            # end FOR
            fid.close()
            loadFlag = True
        # end IF

        # if can't find the fit.txt file for a long time:
        if loadAttempt >= attemptMax:
            sys.exit('Can''t load '+fileName)
        # end IF
    # end WHILE
    return dataVar
# end DEF 
    
def write_text_file(fileName, data):
    fid = open(fileName,'w+')
    # check if data is 1 variable or list/array:
    if (isinstance(data,float) or isinstance(data,int)):
        fid.write(str(data))
    else:
        for j in xrange(0,len(data)):
            fid.write(str(data[j])+"\n")
        # end FOR
    # end IF
    fid.close()
# end DEF
    
def read_txt(textFileName, data_type):
    # new better txt loader, outputs a 1D list of values in the file
    dataVar = []
    fid = open(textFileName,'r')
    for line in fid:
        split_line = line.split(' ')
        for values in split_line:
            if values!='\n':
                if data_type=='int':
                    dataVar += [int(values)]
                elif data_type=='float':
                    dataVar += [float(values)]
                # end IF
            # end IF
        # end FOR
    # end FOR                  
    fid.close()
    return dataVar
# end READTXT
    
def plot_spikes(pop_member, row_len):
    # read the firings file
    firings = read_txt("firings.txt", "int")
    # select times and ids:
    neuron_ids = firings[1::2]
    spike_times = firings[0::2]
    # get max neural sim time in ms
    max_t = max(spike_times)
    # get sample number:
    max_samp = len(spike_times)
    # init EEG array:
    EEG = np.zeros((max_t))
    # get number of neurons:
    # N = max(neuron_ids)
    
    # sum all of the spikes to get the pseudo-EEG:
    for samp in xrange(1,max_samp):
        # Adjust for MATLAB indexing by 1:
        curr_neur_num = neuron_ids[samp]-1
        curr_neur_sign = (pop_member[curr_neur_num * row_len]-0.5)/0.5
        curr_spike_time = spike_times[samp]-1
        EEG[curr_spike_time] = EEG[curr_spike_time] + curr_neur_sign
    # end FOR
        
    # PERFORM SPECTRAL ANALYSIS:

    # split long recordings into 1000 ms epochs:
    epoch_size = 1000
    num_epochs = max_t / epoch_size
    for epoch in xrange(0, num_epochs):
        #    
        epoch_begin = epoch * epoch_size
        epoch_end = epoch_begin + epoch_size
        plb.figure(figsize=(24.0,20.0))
        plb.suptitle('Neural dynamics '+str(epoch_begin)+' to '+str(epoch_end)+' ms')
        time_array = (np.arange(max_t)+1)
        # Spike raster:
        plb.subplot(411)
        plb.title('Spikes')
        plb.xlabel('Time, ms')
        plb.ylabel('Neuron #')
        # find indices of spike_times that are closest to the epoch boundaries:
        time_begin_id,_  = min(enumerate(spike_times), key=lambda x: abs(x[1]-epoch_begin))
        time_end_id,_  = min(enumerate(spike_times), key=lambda x: abs(x[1]-epoch_end))
        # plot dots:
        plb.plot(spike_times[time_begin_id:time_end_id], neuron_ids[time_begin_id:time_end_id],'k.')
        
        # EEG:    
        plb.subplot(412)
        plb.title('Pseudo-EEG')
        plb.plot(time_array[epoch_begin:epoch_end],EEG[epoch_begin:epoch_end])
        plb.axis('tight')
        plb.xlabel('Time, ms')
        plb.ylabel('Spike count')
        
        # Periodogram:
        plb.subplot(413)
        plb.title('Spectral density')
        plb.psd(EEG[epoch_begin:epoch_end], Fs=1000)
        axis_vals = plb.axis()
        plb.axis([0, 100, axis_vals[2], axis_vals[3]])
        
        # Spectrogram:
        plb.subplot(414)
        plb.title('Spectrogram')
        plb.specgram(EEG[epoch_begin:epoch_end], Fs=1000)
        axis_vals = plb.axis()
        plb.axis([0, 1, 0, 100])
        plb.colorbar()
        plb.savefig('neural_dynamics'+str(epoch)+'.png')        
# end plotSpikes
        
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
# end COPYANYTHING  
""" 

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
                 # SNN params:
                 node_types=['tonic_spike'],
                 n_nodes_input=None,
                 n_nodes_hidden=None,
                 n_nodes_output=None,
                 add_bias=True,
                 out_ma_len=20,
                 tau=5.0,
                 weight_epsilon=0.01,
                 max_ticks=1000):
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
        
        # SNN params:
        # node_types - string name(s) of types of spiking neurons to be used for hidden layer
        # n_nodes_input - number of input nodes (not including the bias neuron)
        # n_nodes_hidden - number of hidden nodes (these are the only spiking neurons)
        # n_nodes_output - number of output nodes (CTRNN-type nodes)
        # add_bias - True/False flag that adds/removes bias node from input layer that always has 1.0 state
        # out_ma_len - length of the moving average window that is used to approximate firing of hidden neurons
        # tau - time constant for output neurons (larger values make for more slowly-changing neurons)
        # weight_epsilon - weights below this value are considered equal to 0
        # max_ticks - the number of time steps used to simulate SNN activity
        
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
        
        # SNN stuff:
        self.node_types=node_types
        self.n_nodes_input=n_nodes_input
        self.n_nodes_hidden=n_nodes_hidden
        self.n_nodes_output=n_nodes_output
        self.add_bias=add_bias
        self.out_ma_len=out_ma_len
        self.tau=tau
        self.weight_epsilon=weight_epsilon
        self.max_ticks=max_ticks
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
        n_nodes_hidden = self.n_nodes_hidden
        n_nodes_output = self.n_nodes_output
        
        w_ih = np.zeros((n_nodes_input, n_nodes_hidden), dtype=float)
        w_hh = np.zeros((n_nodes_hidden, n_nodes_hidden), dtype=float)
        w_ho = np.zeros((n_nodes_hidden, n_nodes_output), dtype=float)
        
        pointer = 0
        
        for row in xrange(0, n_nodes_input):
            start_id = row * n_nodes_hidden
            end_id = start_id + n_nodes_hidden
            
            # DEBUG:
            # print "W_IH. Processign row", row
            # print "Start_id =", start_id
            # print "End_id =", end_id
            # print "w_ih[",row,",]=",w_ih[row,]
            # print "pop_member[",start_id,":",end_id,"]", pop_member[start_id:end_id]
            
            
            w_ih[row,] = pop_member[start_id:end_id]
        
        pointer = end_id
        
        # DEBUG
        # print "w_ih=",w_ih
        
        for row in xrange(0, n_nodes_hidden):
            start_id = row * n_nodes_hidden + pointer
            end_id = start_id + n_nodes_hidden

            # DEBUG:
            # print "W_HH. Processign row", row
            # print "Start_id =", start_id
            # print "End_id =", end_id
            # print "w_hh[",row,",]=",w_hh[row,]
            # print "pop_member[",start_id,":",end_id,"]", pop_member[start_id:end_id]
                        
            w_hh[row,] = pop_member[start_id:end_id]
        
        pointer = end_id
        
        # DEBUG
        # print "w_hh=",w_hh
        
        for row in xrange(0, n_nodes_hidden):
            start_id = row * n_nodes_output + pointer
            end_id = start_id + n_nodes_output
            
            # DEBUG:
            # print "W_HO. Processign row", row
            # print "Start_id =", start_id
            # print "End_id =", end_id
            # print "w_ho[",row,",]=",w_ho[row,]
            # print "pop_member[",start_id,":",end_id,"]", pop_member[start_id:end_id]
            
            
            w_ho[row,] = pop_member[start_id:end_id]
        
        # DEBUG:
        # print "w_ho=",w_ho
        
        # DEBUG:
        # print "Pop.member had length",pop_member.shape[0]
        # print "Last weight extracted from it had idx",end_id
        
        if save_flag:
            filename = fname + '_IH.txt'
            self.save_weights_txt(w_ih, filename)
            
            filename = fname + '_HH.txt'
            self.save_weights_txt(w_hh, filename)            
            
            filename = fname + '_HO.txt'
            self.save_weights_txt(w_ho, filename)            
            
        return (w_ih, w_hh, w_ho)        
        
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
        (w_ih, w_hh, w_ho) = self.convert_pop_member(pop_member, save_flag=False)
        
        # update the SNN with weight matrices:
        network.from_matrix(w_ih, w_hh, w_ho, self.node_types)    
            
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
                    """
                    if child1[y]==0: # determine the bit value and flip it
                        child1[y]=1
                    else:
                        child1[y]=0
                    # end IF
                    """
                    child1[y]+=np.random.randint(-self.mut_range-1,self.mut_range+1)
                    # Check that result is within bounds:
                    if child1[y]>self.weight_range:
                        child1[y]=self.weight_range
                    if child1[y]<-self.weight_range:
                        child1[y]=-self.weight_range                        
                # end IF
            
                # roll a die for the second child:
                if rd.uniform(0.0,1.0) < self.prob_mut:
                    """
                    if child2[y]==0: # determine the bit value and flip it
                        child2[y]=1
                    else:
                        child2[y]=0
                    # end IF
                    """
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
                    # print "Evaluating",member,"member"
                # end IF
            # end FOR
            
            # store this generation's best and avg fitness:
            best_fit[gen]=max(fitness)
            avg_fit[gen]=mean(fitness)
            if verbose:
                print "Best fit=",best_fit[gen],"Avg.fit=",avg_fit[gen]
    
            # select parents for mating:
            (par_ids, worst_ids) = self.select_parents(fitness)
            # print "Chose parents",par_ids,"Will replace these",worst_ids
            # for idx in xrange(0,len(worst_ids)):
            #    print "Fitness[",worst_ids[idx],"]=",fitness[worst_ids[idx]]
    
            # replace worst with children produced by crossover and mutation:
            (pop,was_changed) = self.produce_offspring(pop, par_ids, worst_ids, was_changed)
            
    
        # end FOR === MAIN CYCLE ===
        if save_results:
            # SAVE 10 BEST:
            best10_ids = fitness.argsort()[-10:][::-1]
            for idx in xrange(0,len(best10_ids)):
                best_name = 'best_weights'+str(0)+str(idx+1)
                self.convert_pop_member(pop[best10_ids[idx]], save_flag=True, fname=best_name)
                #write_weights(best_name, pop[best10_ids[idx]], row_len)
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
            (w_ih, w_hh, w_ho) = self.convert_pop_member(pop[best10_ids[0]], save_flag=False)
        
            # update the SNN with weight matrices:
            best_network.from_matrix(w_ih, w_hh, w_ho, self.node_types)      
            
            # now plot behavior: 
            self.show_behavior(best_network)    
            # os.mkdir('Results')
            # os.chdir('Results')

            # # copy DEMO to RESULTS:
            # shutil.copyfile(currDir+"\\PDSTEP_demo.exe",currDir+'\\Results\\PDSTEP_demo.exe')
            # shutil.copyfile(currDir+"\\PDSTEP_demo.ilk",currDir+'\\Results\\PDSTEP_demo.ilk')
            # shutil.copyfile(currDir+"\\PDSTEP_demo.pdb",currDir+'\\Results\\PDSTEP_demo.pdb')
            # shutil.copyfile(currDir+"\\PDSTEP_demo_x64_debug.pdb",currDir+'\\Results\\PDSTEP_demo_x64_debug.pdb')
            # shutil.copyfile(currDir+"\\glut64.dll",currDir+'\\Results\\glut64.dll')
            # shutil.copyfile(currDir+"\\best_weights01.txt",currDir+'\\Results\\weights.txt')
        
            # # run DEMO file that will export spike data:
            # os.system('PDSTEP_demo.exe')
        
        # plot spiking data and analyze
        # plot_spikes(pop[0], row_len)        
        
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

    def show_behavior(self, network):
        """
        This method plots behavior of output nodes vs. the targets
        """
        
        network.run()
        gens=np.arange(0, network.max_ticks)
        
        # DEBUG:
        # print "Output node 0 history:", network.out_states_history[:,0]
        # print "Output node 1 history:", network.out_states_history[:,1]
            
        for node in xrange(0,network.n_nodes_output):
            plb.figure()
            plb.plot(gens, network.out_states_history[:,node],'r',label='Output node '+str(node))
            target_vector = np.full((network.max_ticks), self.target_values[node])
            plb.plot(gens, target_vector, 'b', label='Target for node '+str(node))
            plb.title('Behavior plot')
            plb.xlabel('Simulation tick')
            plb.ylabel('Node output vs. target')
            plb.legend()
        # plb.savefig('behavior_plot.png')
            plb.show()
        
    # END SHOW_BEHAVIOR
    
    """
    def write_weights(fileName, data, row_len):
        fid = open(fileName,'w+')
        for j in xrange(0,len(data)):
            if m.fmod(j+1,row_len)==0:
                fid.write(str(data[j])+"\n")
            else:
                fid.write(str(data[j])+" ")
        # end FOR
        fid.close()
    """

    
if __name__ == 'main':
    #
    #
    """
    # get current directory path:
    currDir = os.getcwd()
    # create a new, unique folder for the run:
    EXP_ID = "ST-F"
    timeStamp = t.localtime()
    # Experiment ID tag has timestamp a random number (to make each folder have unique name)
    uniq = int(rd.random()*1000)
    folderName = EXP_ID+"-"+str(timeStamp.tm_mday)+"-"+str(timeStamp.tm_mon)\
                +"-"+str(timeStamp.tm_year)+"-"+ str(timeStamp.tm_hour)\
                +"-"+str(timeStamp.tm_min)+"-"+str(timeStamp.tm_sec)+"-"+str(uniq)  

    # # SNN parameters:
    num_input = 3
    num_hidden = 24
    num_output = 3
    add_bias = True
    # GA parameters:
    # Each individual in the population contains weights that connect inputs to 
    # hidden, hidden to hidden, and hidden to output:    
    len_indv = num_input * num_hidden + num_hidden * num_hidden + num_hidden * num_output
    if add_bias:
        len_indv += num_hidden
        # if there is a bias input neuron (which has always state = 1), need to 
        # provide synaptic weights connecting this bias neuron with all of the 
        # hidden neurons
    pop_size = 50
    prob_xover = 1.0
    prob_mut = 0.05
    prob_survival=0.05
    num_par = 40
    max_gen = 100
    weight_range=50.0 # weights will be integers within [-weight_range; +weight_range]
    
    # Initialize genetic algorithm object:
    genetic_algorithm = GeneticAlgorithm(len_indv=len_indv,
                                         pop_size=pop_size,
                                         num_par=num_par,
                                         prob_mut=prob_mut, 
                                         prob_xover=prob_xover,
                                         prob_survival=prob_survival,
                                         max_gen=max_gen,
                                         weight_range=weight_range) 
    
    genetic_algorithm.evolve(verbose=True, save_results=True)
    """
    print "Class GeneticAlgorithm"
