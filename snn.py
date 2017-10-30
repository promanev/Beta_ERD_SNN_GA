""" 
Package with some classes to simulate spiking neural nets.
This version is for experiment 9.3: switching outputs mid-simulation. 
A cortical input added to the inputs. Weights connecting cortex with the hidden 
layer are evolved too (for the lack of better judgement on what values should be
hardcoded for them!)
    
"""

### IMPORTS ###

import numpy as np
np.set_printoptions(precision=3)
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
                 w_ch=None,
                 w_bgh=None,
                 w_hh=None,
                 w_ho=None,
                 node_types=['tonic_spike'],
                 n_nodes_input=1,
                 n_nodes_cortex=None,
                 n_nodes_hidden=1,
                 n_nodes_output=1,
                 fired_ids=None,
                 add_bias=True,
                 out_ma_len=10,
                 tau=1.5,
                 weight_epsilon=1e-3,
                 max_ticks=1000,
                 switch_tick=500,
                 bg_nodes=2,
                 ERD_tick=450,
                 ERS_tick=550,
                 bg_flag=False,
                 bg_sync_freq=25,
                 bg_is_sync=False):
        
        # Set instance vars
        self.w_ih                 = w_ih
        self.w_ch                 = w_ch
        self.w_bgh                = w_bgh
        self.w_hh                 = w_hh
        self.w_ho                 = w_ho
        self.node_types           = node_types
        self.n_nodes_input        = n_nodes_input
        self.n_nodes_cortex       = n_nodes_cortex
        self.n_nodes_hidden       = n_nodes_hidden
        self.n_nodes_output       = n_nodes_output
        self.fired_ids            = fired_ids # binary vector containing hidden nodes' spikes (updated every tick of SNN simulation) 
        self.add_bias             = add_bias
        self.out_ma_len           = out_ma_len  # The length of the window for the moving average that averages firing of the hidden neurons for the update of the output neurons
        self.tau                  = tau  # time constant for all output neurons (used to update their state)
        self.weight_epsilon       = weight_epsilon # Min.vlue of a synaptic weight for it to be considered for calculation         
        self.max_ticks            = max_ticks
        self.switch_tick          = switch_tick
        self.bg_nodes             = bg_nodes
        self.ERD_tick             = ERD_tick
        self.ERS_tick             = ERS_tick
        self.bg_flag              = bg_flag
        self.bg_sync_freq         = bg_sync_freq
        self.bg_is_sync           = bg_is_sync


    def convert_nodes(self):
    # This method converts string-formatted names of neuron types into actual functions    
        nt = []
        for fn in self.node_types:
            nt.append(NEURON_TYPES[fn])
        self.node_types = nt
            
    
    #TO DO: Have to take in weights as several matrices for each layer 
    # separately (only feed-forward connnetctions and intra-layer 
    # connections for the hidden layer):         
    def from_matrix(self, w_ih, w_hh, w_ho, w_ch=None, w_bgh=None, node_types=['tonic_spike']):
        """ Constructs a network from weight matrices: 
            w_ih - weights from inputs to hidden
            w_ch - weights from cortex to hidden (defaulted to None to allow backwards compatibility)
            w_bgh - weights from the basal ganglia to hidden (defaulted to None to allow backwards compatibility)
            w_hh - weights from hidden to hidden
            w_ho - weights from hidden to output
            node_types - the number of types should be equal to the number of 
                         hidden nodes (these are the only spiking neurons). 

        """
        # Make sure that the network object has the number of neurons set up:
        if self.n_nodes_input is None:
            self.n_nodes_input = w_ih.shape[0]
        if w_ch is not None:
            if self.n_nodes_cortex is None:
                self.n_nodes_cortex = w_ch.shape[0]
        if w_bgh is not None:
            if self.bg_nodes is None:
                self.bg_nodes = w_bgh.shape[0]        
                
        if self.n_nodes_hidden is None:
            self.n_nodes_hidden = w_hh.shape[0]
        if self.n_nodes_output is None:
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
        if self.w_ih is None:    
            self.w_ih = w_ih
        if self.w_ch is None and w_ch is not None:
            self.w_ch = w_ch
        if self.w_bgh is None and w_bgh is not None:
            self.w_bgh = w_bgh            
        if self.w_hh is None:    
            self.w_hh = w_hh
        if self.w_ho is None:    
            self.w_ho = w_ho
        
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
    def feed(self, inputs, cortical_inputs, bg_inputs=None, verbose=False):
        """
        This function runs the simulation of an SNN for one tick using forward Euler 
        integration method with step 0.5 (2 summation steps). This approach is
        used by Izhikevich (2003).
        self - SNN object that should have all of the neuron types (parameters a, b, c ,d) 
        as well as their v and u values. 
        
        inputs    - real-valued numbers representing values of the input neurons (can be 
                    sensor data or return of the output neurons' states).
        fired_ids - a binary vector containing the binary states of hidden neurons maintained externally.
                    Should be all zeros for the first tick
        cortical_inputs - binary inputs from 2 neurons that output [1,0] for 
                          motor state 1 and [0,1] for motor state 2
        bg_inputs - (optional) inputs from simulated basal ganglia neurons into
                    the hidden layer                  
                    
        """
        
        # Some housekeeping:
        n_nodes_input = self.n_nodes_input
        n_nodes_cortex = self.n_nodes_cortex
        n_nodes_hidden = self.n_nodes_hidden
        bg_nodes = self.bg_nodes
        
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
        
        if self.add_bias:
            inputs = np.hstack((inputs,1.0))          
            n_nodes_input+=1
        # Optional but used in the Vasu & Izquierdo (2017):
        # Pass inputs through sigmoidal activation function:
        inputs = 1.0 / (1.0 + np.exp(-inputs)) 

        # 1a. Process CORTEX->HIDDEN:  
        for i in xrange(0, n_nodes_cortex):
            # DEBUG
            # print "Processing cortical input node #",i,"out of", n_nodes_cortex
            
            for j in xrange(0, n_nodes_hidden):                
#                if self.w_ch[i,j] is not 0.0: # skip the next step if the synaptic connection = 0 for speed
                # if verbose:
                    # print "I[",j,"]=",I[j]
                    # print "self.w_ch has shape:",self.w_ch.shape
                    # print "self.w_ch[",i,",",j,"]=",self.w_ch[i,j]
                    # print "cortical_inputs[",i,"]=",cortical_inputs[i]
                I[j] += self.w_ch[i,j] * cortical_inputs[i]
            
        # 1b. Process INPUTS->HIDDEN:    
        for i in xrange(0, n_nodes_input):
            for j in xrange(0, n_nodes_hidden):
#                if self.w_ih[i,j] is not 0.0: # skip the next step if the synaptic connection = 0 for speed
                I[j] += self.w_ih[i,j] * inputs[i]
        
        # 1c. (OPTIONAL, only if provided) Process BG->HIDDEN:    
        if bg_inputs is not None:    
            # DEBUG:
            # print "BG_inputs are", bg_inputs        
            # print "I before BG inputs",I
            for i in xrange(0, bg_nodes):
                # DEBUG
                # print "Processing input node #",i,"out of", n_nodes_input
                for j in xrange(0, n_nodes_hidden):
                    # DEBUG:
                    # print "Processing hidden node #",j,"out of", n_nodes_hidden
                    # print "bg_inputs[",i,"]=", bg_inputs[i]
                    # print "w_bgh[",i,",",j,"]=",self.w_bgh[i,j]
#                    if bg_inputs[i] is not 0.0:
                    # I[j] += self.w_bgh[i,j] * bg_inputs[i]
                    I[j] += -10.0 * bg_inputs[i]
            # DEBUG:
            # print "After processing BG_inputs, I=",I
        
        
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
        
        # 2a. (OPTIONAL) Now that all spikes that happened on the previous tick are detected,
        # process HIDDEN->HIDDEN:
        if self.w_hh is not None:    
            for i in xrange(0, n_nodes_hidden):
                if self.fired_ids[i]>0:
                    for j in xrange(0, n_nodes_hidden):            
#                        if self.w_hh[i,j] is not 0.0:
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
    
    def run(self, verbose=False):
        """
        Since the output neurons are not spiking and depend on the moving average across some window of
        spikes produced by hidden neurons, this method will run the SNN prescribed number of ticks (max_ticks).
        * This method loops over "feed" method.
        * This method outputs a matrix containing the states of output neurons for the analysis by a fitness function
        * Due to the nature of the current experiment, the outputs serve as inputs for the SNN on the next tick
        * The way the cortical input changes throughout the simulation is encoded here
        
        max_ticks - the number of steps to run the SNN
        switch_tick - the step after the cortical input changes
        """
        # IMPORTANT HARDCODED PARAMETER
        # Integration time steps for updating output neurons:
        integration_steps = 10   
        
        # For simplicity, the original "cortex" will have 2 neurons. The first 
        # will be active throughout the first part of hte simulation, the second 
        # neurons will be active after switching only:
        cortical_inputs = np.array([ [1.0, 0.0],
                                     [0.0, 1.0] ])    
        
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
        
        # Init a vector that will hold all of BG spikes:
        bg_nodes_history = np.zeros((self.max_ticks, self.bg_nodes))
        
        # Main cycle
        for t in xrange(0, self.max_ticks):
            # First, determine what type of cortical input to provide to the SNN:
            if t < self.switch_tick:                
                cortical_input_this = cortical_inputs[0,]
            else:
                cortical_input_this = cortical_inputs[1,]
                
            # Check which state the BG should be in:
            if (t < self.ERD_tick) or (t > self.ERS_tick):
                # BG are SYNC, all nodes fire at the same 
                # time using provided frequency:
                tick_to_spike = int(1000.0/self.bg_sync_freq)    
                # Create bg_inupts:
                if np.mod(t,tick_to_spike)==0:    
                    bg_inputs=np.ones((self.bg_nodes))
                else:
                    bg_inputs=np.zeros((self.bg_nodes))
                # DEBUG:
                # print "Tick",t,"; BG are SYNC"
                # print "BG_inputs =",bg_inputs
                        
            else:
                # BG are DESYNC and are firing in the random mode:
                # 
                # Create random spikes:
                # This creates a very active firing pattern
                # bg_inputs=np.random.randint(2, size=(self.bg_nodes,))    
                # This should create a rather sparse random pattern:
                bg_inputs=np.zeros((self.bg_nodes))    
                for n in xrange(0,self.bg_nodes):
                    if np.random.random() < 0.1:
                        bg_inputs[n]=1.0
                # DEBUG:
                # print "Tick",t,"; BG are DESYNC"
                # print "BG_inputs =",bg_inputs
                    
            bg_nodes_history[t,] = bg_inputs                
            
            # during the first tick the SNN receives the maximal stimulus to bootstrap the neural activity:
            if t==0:
                self.feed(np.ones(self.n_nodes_input), cortical_input_this, bg_inputs, verbose=verbose)
            else:
                self.feed(out_states_history[t-1,], cortical_input_this, bg_inputs, verbose=verbose)
                
            # record the states of the hidden neurons after this tick has been simulated:
            hid_states_history[t,] = self.fired_ids
            
            # now approximate the output neurons. Ideally, last self.out_ma_len ticks 
            # should be used, but dirung the first ticks the window is shorter:
            if (t+1)<self.out_ma_len:
                window_len = t+1
            else:
                window_len = self.out_ma_len
   
            # init a local vector that will store approximated states of hidden neurons 
            # (will be used to update output neurons):
            hidden_avg_states = np.zeros(self.n_nodes_hidden)
            
            # there is no point of averaging neurons' states on the first tick as there is no history and 
            # averaging function over [0,0] produces nan:
            if t>0:
                for i in xrange(0, self.n_nodes_hidden):
                    hidden_avg_states[i] = np.mean(hid_states_history[t+1-window_len:t, i])
            
                    # if verbose:
                        # print "Neuron",i," avg. fir.rate =",hidden_avg_states[i]
                        
            # now update states of output neurons:
            for i in xrange(0,self.n_nodes_output):
                
                # DEBUG:
                # if verbose:    
                    # print "Calculating influence on output neuron #", i
                
                # first calculate the sum of influences of all hidden neurons on this output neuron:
                influence = 0.0    
                for j in xrange(0,self.n_nodes_hidden):
                    
                    # if verbose:
                    # DEBUG:
                        # print "Avg. state of hidden neuron #",j,"is", hidden_avg_states[j]
                        # print "Weight w_ho[",j,",",i,"] =", self.w_ho[j,i]
                    
                    influence += self.w_ho[j,i] * hidden_avg_states[j]
                   
                    # if verbose:
                    # DEBUG:
                        # print "Updated influence =", influence
                    
                # numerically integrating using forward Euler method with 2 time steps:                
                for h in xrange(0, integration_steps):
                    out_state[i] += (1.0 / float(integration_steps)) * (1.0 / self.tau) * (-out_state[i] + influence) 

            # if verbose:
                # print "Output neurons' states:", out_state
                
            out_states_history[t,] = out_state
           
        # end MAIN CYCLE 
        self.out_states_history = out_states_history
        self.hid_states_history = hid_states_history
        self.bg_nodes_history = bg_nodes_history
        
        return self
    # end RUN
                
                
# DON'T NEED THIS?
if __name__ == '__main__':
    print "SNN class main"
