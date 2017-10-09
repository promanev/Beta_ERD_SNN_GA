""" Package with some classes to simulate spiking neural nets.
"""

### IMPORTS ###

import numpy as np
# import graphviz_plot as gpv
np.seterr(over='ignore', divide='raise')

# Local class
class CustomStructure:
    def __init__(self,**kwds):
        self.__dict__.update(kwds)

inf = float('inf')
sqrt_two_pi = np.sqrt(np.pi * 2)

# Neuron types:
# Apperantly, this is a version of a regular spiking neuron with fast recovery (due to low d - reset of the leaking variable u)
# For reference, see Izhikevich(2003), Figure 2, upper right corner.
def fast_spike():
    params = CustomStructure(a=0.1, b=0.2, c=-65.0, d=2.0)
    return params

def tonic_spike():
    params = CustomStructure(a=0.02, b=0.2, c=-65.0, d=6.0)
    return params

def phasic_spike():
    params = CustomStructure(a=0.02, b=0.25, c=-65.0, d=6.0)
    return params

def tonic_burst():
    params = CustomStructure(a=0.02, b=0.2, c=-50.0, d=2.0)
    return params

def phasic_burst():
    params = CustomStructure(a=0.02, b=0.25, c=-55.0, d=0.05)
    return params

def mixed():
    params = CustomStructure(a=0.02, b=0.2, c=-55.0, d=4.0)
    return params

def fq_adapt():
    params = CustomStructure(a=0.01, b=0.2, c=-65.0, d=8.0)
    return params

def class1():
    params = CustomStructure(a=0.02, b=-0.1, c=-55.0, d=6.0)
    return params

def class2():
    params = CustomStructure(a=0.2, b=0.26, c=-65.0, d=0.0)
    return params

def spike_lat():
    params = CustomStructure(a=0.02, b=0.2, c=-65.0, d=6.0)
    return params

def subthresh():
    params = CustomStructure(a=0.05, b=0.26, c=-60.0, d=0.0)
    return params

def reson():
    params = CustomStructure(a=0.1, b=0.26, c=-60.0, d=-1.0)
    return params

def integr():
    params = CustomStructure(a=0.02, b=-0.1, c=-55.0, d=6.0)
    return params

def rebound_spike():
    params = CustomStructure(a=0.03, b=0.25, c=-60.0, d=4.0)
    return params

def rebound_burst():
    params = CustomStructure(a=0.03, b=0.25, c=-52.0, d=0.0)
    return params
# threshold variability:
def thresh_var():
    params = CustomStructure(a=0.03, b=0.25, c=-60.0, d=4.0)
    return params

def bistab():
    params = CustomStructure(a=1.0, b=1.5, c=-60.0, d=0.0)
    return params
# depolarizng after-potential
def dap():
    params = CustomStructure(a=1.0, b=0.2, c=-60.0, d=-21.0)
    return params
# accomodation:
def accom():
    params = CustomStructure(a=0.02, b=1.0, c=-55.0, d=4.0)
    return params
# inhibition-induced spiking:
def ii_spike():
    params = CustomStructure(a=-0.02, b=-1.0, c=-60.0, d=8.0)
    return params    
# inhibition-induced bursting:
def ii_burst():
    params = CustomStructure(a=-0.026, b=-1.0, c=-45.0, d=0.0)
    return params    
### CONSTANTS ###

NEURON_TYPES = {
    'fast_spike': fast_spike,   
    'tonic_spike': tonic_spike,
    'phasic_spike': phasic_spike,    
    'tonic_burst': tonic_burst,    
    'phasic_burst': phasic_burst,    
    'mixed': mixed,
    'fq_adapt': fq_adapt,
    'class1': class1,
    'class2': class2,
    'spike_lat': spike_lat,
    'subthresh': subthresh,
    'reson': reson,
    'integr': integr,
    'rebound_spike': rebound_spike,
    'rebound_burst': rebound_burst,
    'thresh_var': thresh_var,
    'bistab': bistab,
    'dap': dap,
    'accom': accom,
    'ii_spike': ii_spike,
    'ii_burst': ii_burst,
    None: tonic_spike    
        
}

### CLASSES ### 

class SpikingNeuralNetwork(object):
    """ A neural network. Can have recursive connections.
    """
    
    def __init__(self,
                 w_ih=None,
                 w_hh=None,
                 w_ho=None,
                 node_types=['tonic_spike'],
                 n_nodes_input=1,
                 n_nodes_hidden=1,
                 n_nodes_output=1,
                 fired_ids=None,
                 add_bias=True,
                 out_ma_len=10,
                 tau=1.5,
                 weight_epsilon=1e-3,
                 max_ticks=1000):
        # Set instance vars
        self.w_ih                 = w_ih
        self.w_hh                 = w_hh
        self.w_ho                 = w_ho
        self.node_types           = node_types
        self.n_nodes_input        = n_nodes_input
        self.n_nodes_hidden       = n_nodes_hidden
        self.n_nodes_output       = n_nodes_output
        self.fired_ids            = fired_ids # binary vector containing hidden nodes' spikes (updated every tick of SNN simulation) 
        self.add_bias             = add_bias
        self.out_ma_len           = out_ma_len  # The length of the window for the moving average that averages firing of the hidden neurons for the update of the output neurons
        self.tau                  = tau  # time constant for all output neurons (used to update their state)
        self.weight_epsilon       = weight_epsilon # Min.vlue of a synaptic weight for it to be considered for calculation         
        self.max_ticks            = max_ticks
        # convert node names into functions:
        # self.convert_nodes()



    def convert_nodes(self):
    # This method converts string-formatted names of neuron types into actual functions    
        nt = []
        for fn in self.node_types:
            nt.append(NEURON_TYPES[fn])
        self.node_types = nt
            
    
    #TO DO: Have to take in weights as several matrices for each layer 
    # separately (only feed-forward connnetctions and intra-layer 
    # connections for the hidden layer):         
    def from_matrix(self, w_ih, w_hh, w_ho, node_types):
        """ Constructs a network from weight matrices: 
            w_ih - weights from inputs to hidden
            w_hh - weights from hidden to hidden
            w_ho - weights from hidden to output
            node_types - the number of types should be equal to the number of 
                         hidden nodes (these are the only spiking neurons). 

        """
        if self.n_nodes_input == None:
            self.n_nodes_input = w_ih.shape[0]
        if self.n_nodes_hidden == None:
            self.n_nodes_hidden = w_hh.shape[0]
        if self.n_nodes_output == None:
            self.n_nodes_output = w_ho.shape[1]
        
        # Check if all of the hidden neurons are the same type:
        if len(node_types)==1:    
            self.node_types = node_types * self.n_nodes_hidden
        # otherwise node_types should be of the same length as the number of hidden neurons
        else:    
            self.node_types = node_types 
            
        # make sure that node types are converted from str into function calls:
        # print type(self.node_types)    
        self.convert_nodes()    

        # init weight matrices:
        if self.w_ih == None:    
            self.w_ih  = w_ih
        if self.w_hh == None:    
            self.w_hh  = w_hh
        if self.w_ho == None:    
            self.w_ho  = w_ho
        
        # init variables u and v:    
        self.v = np.zeros(self.n_nodes_hidden)
        self.u = np.zeros(self.n_nodes_hidden)
        # Provide initial v and u values based on the each neuron type: 
        for i in xrange(0, self.n_nodes_hidden):
            params = self.node_types[i]()
            self.v[i] = params.c
            self.u[i] = self.v[i] * params.b
            
        # initialize the vector that will store binary representation of spikes:
        self.fired_ids = np.zeros(self.n_nodes_hidden)
        
        return self
             
    #TO DO: Needs to accomodate: a) inputs and outputs are regular neurons, 
    # only hidden are spiking (like in Candadai Vasu and Izquierdo, 2017), 
    # b) there are separate weight matrices, c) the state of the output neuron 
    # is passed through a moving average (Candadai Vasu and Izquierdo, 2017), 
    # d) there is optional bias neuron feeding into hidden layer              
    def feed(self, inputs):
        """
        This function runs the simulation of an SNN for one tick using forward Euler 
        integration method wtih step 0.5 (2 summation steps). This approach is
        used by Izhikevich (2003).
        self - SNN object that should have all of the neuron types (parameters a, b, c ,d) 
        as well as their v and u values. 
        
        inputs    - real-valued numbers representing values of the input neurons (can be 
                    sensor data or return of the output neurons' states).
        fired_ids - a binary vector containing the binary states of hidden neurons maintained externally.
                    Should be all zeros for the first tick
        """
        
        # Some housekeeping:
        n_nodes_input = self.n_nodes_input
        n_nodes_hidden = self.n_nodes_hidden
        
        # minimum weight that is considered for calculation:
        weight_epsilon = self.weight_epsilon 
        
        # node types vector:
        node_types = self.node_types
        # vector with membrane potentials of all simulated neurons (hidden and output?):
        v = self.v
        # vector with recovery variables:
        u = self.u
        
        # Now that the previous state of hidden neurons was used recorded into history
        # (outside, in the "run" method), it needs to be reset for spikes that could be detected 
        # during this tick:
        self.fired_ids = np.zeros(n_nodes_hidden)        
        
        # Input vector that contains all of the influences on the hidden neurons this time step
        # Note: only hidden neurons and external inputs (processed by the input layer) can change this vector. 
        # Output neurons cannot do this as it is assumed that there are no recurrent connections.
        I = np.zeros(n_nodes_hidden)
        
        # DEBUG:
        # print "Network received inputs:",inputs
        if self.add_bias:
            
            # DEBUG:
            # print "Inputs =", inputs
            
            inputs = np.hstack((inputs,1.0))
            
            # DEBUG:
            # print "...but since there is a bias node, I am adding a 1.0 in the end:", inputs
            
            n_nodes_input+=1
        # Optional but used in the Vasu & Izquierdo (2017):
        # Pass inputs through sigmoidal activation function:
        inputs = 1.0 / (1.0 + np.exp(-inputs)) 
        
        # DEBUG:
        # print "Inputs after activation function was applied:", inputs    
            
        # 1. Process INPUTS->HIDDEN:    
        for i in xrange(0, n_nodes_input):
            # DEBUG
            # print "Processing input node #",i,"out of", n_nodes_input
            for j in xrange(0, n_nodes_hidden):
                # DEBUG:
                # print "Processing hidden node #",j,"out of", n_nodes_hidden
                # print "w_ih has shape", self.w_ih.shape
                if self.w_ih[i,j] > weight_epsilon: # skip the next step if the synaptic connection = 0 for speed
                    I[j] += self.w_ih[i,j] * inputs[i]
        # DEBUG
        # print "I after all inputs were processed:"
        # print I

        
        # 2. Detect spikes in the HIDDEN layer (according to membrane potential values reached 
        # on the previous tick) and update the fired_ids vector.                    
        for i in xrange(0, n_nodes_hidden):
            if v[i]>30.0:
                
                # DEBUG:
                # print "Registered a spike in",i,"-th neuron, v[",i,"]=",v[i]   
                
                # Record these data for export out of the function:
                self.fired_ids[i] = 1
                # fetch this node's params:
                params = node_types[i]()  
                v[i] = params.c
                u[i]+= params.d        
        
        # 3. Now that all spikes that happened on the previous tick are detected,
        # process HIDDEN->HIDDEN:
        for i in xrange(0, n_nodes_hidden):
            if self.fired_ids[i]>0:
                for j in xrange(0, n_nodes_hidden):            
                    if self.w_hh[i,j] > weight_epsilon:
                        I[j] += self.w_hh[i,j] # no need to multiply by the state of i-th hidden neuron because it's = 1
        
        # DEBUG:                
        # print "I after all previous hid->hid spikes were processed:"
        # print I
        
        # 4. Update u and v of all of the simulated neurons (hidden only) based on the 
        # calculated I that accounts for the inputs and spikes on hidden neurons that took place 
        # on the previous tick:
        for i in xrange(0, n_nodes_hidden):
            # fetch this node's params:
            params = node_types[i]()                
                
            # Numerical integration using forward Euler method wiht step 0.5 for differential equations governing v and u:
            for tick in xrange(0,2):
                
                # DEBUG:
                # print "Before integrating. v[",adj_i,"]=",v[adj_i]
                # print "Member 1: 0.04*v**2=",0.04 * v[adj_i] ** 2
                # print "Member 2: 5*v=",5 * v[adj_i]
                # print "Member 3: -u=",- u[adj_i]
                # print "Member 4: I",I[adj_i]
                # print "All together =",0.5 * ( 0.04 * v[adj_i] ** 2 + 5 * v[adj_i] + 140 - u[adj_i] + I[adj_i])
                
                v[i] += 0.5 * ( 0.04 * v[i] ** 2 + 5 * v[i] + 140 - u[i] + I[i])
                
                # DEBUG:
                # print "After integrating. v[",adj_i,"]=",v[adj_i]
                
            u[i] += params.a * (params.b * v[i] - u[i]) # It's unclear from Izhikevich's code if u should also updated in two steps or if it's updated once, after v was updated
            
            # DEBUG:
            # print "v[",adj_i,"]=",v[adj_i],"; u[",adj_i,"]=",u[adj_i]
            # print "Neuron #",i,"; v=",v[adj_i]
        
        # DEBUG:
        # print "Fired_ids =", self.fired_ids
        
        # 4. Return ALL pertinent variables:
        self.v = v
        self.u = u
        
        return self
    
    def run(self):
        """
        Since the output neurons are not spiking and depend on the moving average across some window of
        spikes produced by hidden neurons, this method will run the SNN prescribed number of ticks (max_ticks).
        * This method loops over "feed" method.
        * This method outputs a matrix containing the states of output neurons for the analysis by a fitness function
        * Due to the nature of the current experiment, the outputs serve as inputs for the SNN on the next tick
        
        max_ticks - the number of steps to run the SNN
        """
        # IMPORTANT HARDCODED PARAMETER
        # Integration time steps for updating output neurons:
        integration_steps = 4    
        
        # init the matrix that will keep the states of output neurons:
        out_states_history = np.zeros((self.max_ticks, self.n_nodes_output))  # Each row represents the states of output neurons during the previous tick  
        
        # init the matrix that will keep the binary states of hidden neurons 
        # (to compute the states of output neurons using the moving average):
        hid_states_history = np.zeros((self.max_ticks, self.n_nodes_hidden))
        
        # init a vector that will hold dynamically changing values of the output neurons:
        
        # Option 1: Output neurons are initialized with all zeros - mgiht lead to SNN not firing at all?    
        # out_state = np.zeros(self.n_nodes_output)
        
        # Option 2: Output neurons are initialized with all ones - mgiht help SNN to bootstrap internal activity?    
        out_state = np.ones(self.n_nodes_output)
        
        # Main cycle
        for t in xrange(0, self.max_ticks):
            # during the first tick the SNN receives the maximal stimulus to bootstrap the neural activity:
            if t==0:
                self.feed(np.ones(self.n_nodes_input))
            else:
                self.feed(out_states_history[t-1,])
                
            # record the states of the hidden neurons after this tick has been simulated:
            hid_states_history[t,] = self.fired_ids
            
            # DEBUG:
            # print "Fired_ids:", self.fired_ids
            # print "Recorded in the history var:", hid_states_history[t,]
            
            # now approximate the output neurons. Ideally, last self.out_ma_len ticks 
            # should be used, but dirung the first ticks the window is shorter:
            if (t+1)<self.out_ma_len:
                window_len = t+1
            else:
                window_len = self.out_ma_len
                
            # DEBUG:
            # print "Avging window is", self.out_ma_len
            # print "...but using window", window_len,"because the tick is",t
                
            # init a local vector that will store approximated states of hidden neurons 
            # (will be used to update output neurons):
            hidden_avg_states = np.zeros(self.n_nodes_hidden)
            
            # DEBUG:
            # print "Init the vector for averaged hidden neurons' states:",hidden_avg_states
            
            # there is no point of averaging neurons' states on the first tick as there is no history and 
            # averaging function over [0,0] produces nan:
            if t>0:
                for i in xrange(0, self.n_nodes_hidden):
                    hidden_avg_states[i] = np.mean(hid_states_history[t+1-window_len:t, i])
            
                    # DEBUG:
                    # print "Avging hidden neuron #",i
                    # print "Looking up its states over the period [",t+1-window_len,",",t,"]"
                    # print "Its history of firing:",hid_states_history[t+1-window_len:t, i]
                    # print "Therefore, its avg. firing rate =",hidden_avg_states[i]
                
            # now update states of output neurons:
            for i in xrange(0,self.n_nodes_output):
                
                # DEBUG:
                # print "Calculating influence on output neuron #", i
                
                # first calculate the sum of influences of all hidden neurons on this output neuron:
                influence = 0.0    
                for j in xrange(0,self.n_nodes_hidden):
                    
                    # DEBUG:
                    # print "Avg. state of hidden neuron #",j,"is", hidden_avg_states[j]
                    # print "w_ho[",j,",",i,"] =", self.w_ho[j,i]
                    
                    influence += self.w_ho[j,i] * hidden_avg_states[j]
                   
                    # DEBUG:
                    # print "Updated influence =", influence
                    
                # numerically integrating using forward Euler method with 2 time steps:
                
                # DEBUG:
                # print "out_state[",i,"] before num.integrating =", out_state[i]
                
                for h in xrange(0, integration_steps):
                    out_state[i] += (1.0 / float(integration_steps)) * (1.0 / self.tau) * (-out_state[i] + influence) 
                
                    # DEBUG
                    # print "Intrg.step", h, "out_state[",i,"] =", out_state[i]
                    
            out_states_history[t,] = out_state
           
        # end MAIN CYCLE 
        self.out_states_history = out_states_history
        self.hid_states_history = hid_states_history
        
        return self
    # end RUN

    
    """
    This function should output what the network is doing: spikes and/or membrane potentials 
    as well as printing the cm that is used (to make sure that there are all vital connections 
    btw layers)
    """
    # plot_spikes = True, plot_EEG = False, plot_spectrogram = False,
    def plot_behavior(self, filename, save_cm = False, save_spikes = False, save_v = False, add_bias=False):
        # Function that will run the SNN and record its spikes. 
        # Optional: 
        # 1. Create pseudo-EEG and plot it
        # 2. Create spectrogram and plot it
        # 3. Save spikes in a text file, plots as .png
        
        # Imports:
        import pylab as plb
        
        # Some housekeeping:
        # n_nodes_all = self.num_nodes()
        n_nodes_input = self.n_nodes_input
        n_nodes_output = self.n_nodes_output
        n_nodes_hidden = self.n_nodes_hidden
        
        # get connectivity matrix:
        cm = self.cm
        
        # (Optional) Remove unnecessary connection weights:
        # network.condition_cm()
        
        # number of simulation steps:
        max_t = 1000
        # Array to sum all of the spikes of output neurons (to estimate the firing rates):
        firings = np.zeros(n_nodes_output)
        # Array to record all of the spikes of hidden and output neurons each tick in binary format:
        spikes = np.zeros((max_t, n_nodes_hidden + n_nodes_output))
        # Array to record all of the spikes of hidden and output neurons each tick in [neuron_id, tick] format:
        
        # Array with first inputs:
        first_input = np.ones(n_nodes_input) 

        # Save connection matrix:
        if save_cm:
            np.savetxt(filename+'_cm.txt', cm, fmt = '%3.3f')
        
        fired_this_tick = np.zeros(1)
        ticks = np.zeros(1)
        
        # Create the matrix to hold all of the v's:
        if save_v:
            archive_v = np.zeros((max_t, n_nodes_hidden + n_nodes_output))    
        
        # Run the simulation:
        for tick in xrange(0, max_t):
            # On the first tick, feed the SNN with preset inputs. 
            # During all other ticks, feed the SNN with spikes from output nodes:
            if tick == 0:    
                self = self.feed(first_input, add_bias=add_bias)
            else:
                self = self.feed(outputs, add_bias=add_bias)
            # Grab the output
            outputs = self.fired_ids[-n_nodes_output:]
            
            # save the current state:
            if save_v:    
                archive_v[tick] = self.v
                
            # record the state of the hidden and output neurons:
            spikes[tick] = self.fired_ids  
            
            for temp_idx2 in xrange(0, n_nodes_hidden + n_nodes_output):
                if spikes[tick, temp_idx2] == 1.0:
                    fired_this_tick = np.append(fired_this_tick, float(temp_idx2))
                    ticks = np.append(ticks, float(tick))
            
            # add new spikes fired on output neurons:
            for out_idx in xrange(0, n_nodes_output):
                firings[out_idx] += outputs[out_idx] 
                
        # Save the archive of v's:
        if save_v:
            np.savetxt(filename+'_v.txt', archive_v, fmt = '%8.3f')        
                
        # Print the firing rates:
        for out_idx in xrange(0, n_nodes_output):
            print "Output neuron #", out_idx, "has firing rate =", firings[out_idx] 
        
        # plot spikes only if there are spikes:
        if np.sum(ticks) > 0.0:    
            # Record spikes as a plot and .txt file:
            plb.figure(figsize=(12.0,10.0))
            # plb.figure()
            # time_array = np.arange(max_t)
            plb.title('Spikes')
            plb.xlabel('Time, ms')
            plb.ylabel('Neuron #')
            plb.plot(ticks, fired_this_tick,'k.')
            fig1 = plb.gcf()
            axes = plb.gca()
            axes.set_ylim([0,n_nodes_hidden + n_nodes_output])
            axes.set_xlim([0,max_t])
            plb.show()          
        
            if save_spikes:
                fig1.savefig(filename+"_spikes.png", dpi=300) 
        else:
            print "No spikes!"
            
        # Optional: 
        # 1. Create pseudo-EEG and plot it
        # 2. Create spectrogram and plot it
        # 3. Save spikes in a text file, plots as .png
                
                
# DON'T NEED THIS?
if __name__ == '__main__':
    # import doctest
    # doctest.testmod(optionflags=doctest.ELLIPSIS)
    # a = SpikingNeuralNetwork().from_matrix(np.array([[0,0,0],[0,0,0],[1,1,0]]))
    # print a.cm_string()
    # print a.feed(np.array([1,1]), add_bias=False)
    print "SNN class main"
