import random
import numpy as np
import math

class Network(object):
    def __init__(self, sizes):
        """sizes is an array where the layer sizes are listed"""
        # Now we will create the biases and weights accoring to the sizes
        self.sizes = sizes
        self.L = len(sizes)
        #self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.weightsMatrix = [np.random.randn(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
        #self.biasesMatrix = [np.random.randn(y,1) for y in sizes[1:]]
        self.biasesMatrix = []
        for s in range(1, len(sizes)):
            self.biasesMatrix.append(np.random.randn(sizes[s],1))
        #Now time to reformat the biasesMatrix to arrays because we can't use 4x1
        # arrays or else the feedforward method breaks
        for i in range(len(self.biasesMatrix)):
            self.biasesMatrix[i] = toArray(self.biasesMatrix[i])
            
    def setWeightsAndBiases(self, weights, biases):
        self.weightsMatrix = weights[:]
        self.biasesMatrix = biases[:]
       
    def refreshEpoch(self):
        try:
            del self.errors_over_epoch[:]
            del self.activations_over_epoch[:]
            del self.activationsMatrix[:]
            del self.Zs[:]
            del self.ErrorsMatrix[:]
            del self.epoch_summary[:]
            del self.num_correct_in_epoch
        except:
            pass
        self.errors_over_epoch = []
        self.activations_over_epoch = []
        self.activationsMatrix = []
        self.Zs = []
        self.ErrorsMatrix = []
        self.epoch_summary = []
        """Epoch summary should be a list of tuples (size 4)
        List:
          Tuple:
            Vector - Input
            Vector - Expected output
            Vector - Generated output
            Boolean - True or false whether we got it right or not
        """
        self.num_correct_in_epoch = 0        
    
    def learn(self, num_epochs, training_data, eta, output_type):
        """I don't think we need stochastic gradient descent since it's only
        4 pixels at a time, so there aren't many examples we can use to train
        the network..., so we're just going to use a non-stochastic gradient
        descent"""
        """Training data will be of the form of tuples of vectors where the
        first tuple is the activation, and the second tuple is the expected
        output of the network"""
        
        for epoch in range(num_epochs):
            self.refreshEpoch()
            random.shuffle(training_data)
            for training_data_index in range(len(training_data)):
                #Now we should feedforward and such
                batch_pair = training_data[training_data_index]
                
                self.activationsMatrix = [batch_pair[0]]
                self.Zs = []
                self.Zs = [np.zeros(s) for s in self.sizes[1:]]
                
                #Feedforward algorithm
                self.feedforward()
                
                #Compute the error for the last layer, L
                self.ErrorsMatrix = [np.zeros(s) for s in self.sizes[1:]]
                Y = batch_pair[1]
                #Okay so we're screwing up right here on epoch #2...?
                #This next part should fix our typing issue
                for i in range(len(self.Zs)):
                    self.Zs[i] = np.array(self.Zs[i])
                
                costDerivative = self.costDerivative(self.activationsMatrix[-1], Y)
                self.ErrorsMatrix[-1] = np.multiply(costDerivative, sigmoid_prime(self.Zs[-1]))
                
                #Backpropogation algorithm
                self.backpropogate()
                
                self.errors_over_epoch.append(self.ErrorsMatrix)
                self.activations_over_epoch.append(self.activationsMatrix)
                
                #Now we ask if the network got the example correct
                self.evaluateSingle(batch_pair)
                
            #Now we want to use gradient descent
            self.gradientDescent(eta, training_data, epoch)
                
            #Now we should evaluate the network if I can figure out how to do that
            self.evaluateEpoch(epoch, len(training_data), output_type, num_epochs)            
    
    def costDerivative(self, A, Y):
        return A - Y
    
    def evaluateSingle(self, batch_pair):
        """
        Here we are going to generate the "self.epoch_summary" variable
          and record whether we got the example problem correct or not.
        """
        last_activations = self.activationsMatrix[-1]
        #This should concatenate the last layer's activations onto the tuple
        batch_triple = batch_pair + tuple([last_activations])
        theoretical_activations = batch_pair[1]
        if last_activations.shape[0] == 1:
            last_activations = toArray(last_activations)   
        max_index_theoretical = np.argmax(theoretical_activations)
        max_index_experimental = np.argmax(last_activations)
        bool_maxes_match = max_index_theoretical == max_index_experimental
        batch_quadruple = batch_triple + tuple([bool_maxes_match])
        if bool_maxes_match:
            self.num_correct_in_epoch += 1
        self.epoch_summary.append(batch_quadruple)
    
    def gradientDescent(self, eta, training_data, epoch):
        multiplier = -eta/len(training_data)
        for i in range(0, self.L - 1):
            BeforeWeights = self.weightsMatrix[i]
            Summation = 0
            for x in range(len(training_data)):
                errors = to1xNMatrix(self.errors_over_epoch[x][i]).transpose()
                activations = to1xNMatrix(self.activations_over_epoch[x][i])                 
                Summation += np.matmul(errors, activations)
            ChangeInWeights = multiplier * Summation
            AfterWeights = BeforeWeights + ChangeInWeights
            self.weightsMatrix[i] = AfterWeights
            
            BeforeBiases = self.biasesMatrix[i]
            Summation = 0
            for x in range(len(training_data)):
                Summation += self.errors_over_epoch[x][i]
            ChangeInBiases = multiplier * Summation
            AfterBiases = BeforeBiases + ChangeInBiases
            self.biasesMatrix[i] = toArray(AfterBiases)        
    
    def backpropogate(self):
        for i in range(self.L-2, 2, -1):
            left_side = np.matmul(self.weightsMatrix[i + 1].transpose(), self.ErrorsMatrix[i + 1])
            self.ErrorsMatrix[i] = np.multiply(left_side, sigmoid_prime(self.Zs[i]))
        
    def feedforward(self):
        #So self.L is 4, , meaning there are 4 layers which is good
        #print(self.weightsMatrix[-1].shape)
        #print(self.biasesMatrix[-1].shape)
        for i in range(0, self.L - 1):
            self.Zs[i] = np.matmul(self.weightsMatrix[i], self.activationsMatrix[i]) + self.biasesMatrix[i]
            nextActivation = sigmoid(self.Zs[i])
            if nextActivation.shape[0] == 1:
                #print('we tried the transpose')
                nextActivation = nextActivation.transpose()
                nextActivation = toArray(nextActivation)
                #print(' A: {0}'.format(nextActivation.shape))
            self.activationsMatrix.append(nextActivation)
            #print('{0} > {1} * {2} + {3} > {4}'.format(i, self.weightsMatrix[i].shape, self.activationsMatrix[i].shape, self.biasesMatrix[i].shape, self.activationsMatrix[i+1].shape))
            
    def evaluateEpoch(self, epoch, num_trained, output_type, n_epochs):
        """Currently we're working on debugging the neural network and why it really
        isn't learning per-say... Or at least it's not being evaluated correctly.
        We should probably always compare it to one specific example"""
        performance = self.num_correct_in_epoch / num_trained * 100.0
        if output_type == 'all' or (output_type == 'last' and epoch == n_epochs-1):
            print('Epoch {0}: {1:.5g}% => {2} / {3}'.format(epoch+1, performance, self.num_correct_in_epoch, num_trained))
        
    def endprinting(self, array1, array2):
        """This should format our output nicely"""
        for n in range(len(array1)):
            print('{0:.4g}   {1:.4g}'.format(array1[n], array2[n]))
        print('\n')
        
    def inputOutput(self, *io):
        if len(io) == 1:
            """Then we just need to return the output activations"""
            input_activations = io[0]
            self.refreshEpoch()
            self.activationsMatrix = [input_activations]
            self.Zs = []
            self.Zs = [np.zeros(s) for s in self.sizes[1:]]            
            self.feedforward()
            return self.activationsMatrix[-1]
        elif len(io) == 2:
            """Then we should return the output activations and whether it got it right or wrong"""
            input_activations = io[0]
            expected_output_activations = io[1]
            self.refreshEpoch()
            self.activationsMatrix = [input_activations]
            self.Zs = []
            self.Zs = [np.zeros(s) for s in self.sizes[1:]]            
            self.feedforward()
            self.evaluateSingle(tuple([input_activations, expected_output_activations]))
            if self.num_correct_in_epoch == 1:
                return (self.activationsMatrix[-1], True)
            else:
                return (self.activationsMatrix[-1], False)                
        else:
            raise ValueError('Can only recieve maximum 2 input parameters')
        

"""These two functions will output arrays if arrays are inputted"""
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
    #return ReLU(z)
def sigmoid_prime(z):
    return sigmoid(z)*(1.0 - sigmoid(z))
    #return ReLU_prime(z)
def ReLU(z):
    #Anything less than 0 will become a 0
    try:
        z = toArray(z)
    except:
        pass
    return abs(z * (z > 0))     
def ReLU_prime(z):
    #This will turn anything >= 0 into a 1 and anything < 0 into a 0
    try:
        z = toArray(z)
    except:
        pass
    ret = (z > 0) / z * z
    for i in range(len(ret)):
        if math.isnan(ret[i]):
            ret[i] = 0
    #float_formatter = lambda x: "%.2f" % x
    #np.set_printoptions(formatter={'float_kind':float_formatter})    
    #print('{0} => {1}'.format(z, ret)) #This might lag me a bit but I have to try
    return ret

"""Now conversion functions to and from arrays and matrices of 1 dimension"""
def toArray(m):
    if isinstance(m, np.ndarray): #and (m.shape[0] == 1 or m.shape[1] == 1):
        return np.squeeze(np.asarray(m))
    else:
        raise ValueError('Must be of size 1xN or Nx1 matrix')
def to1xNMatrix(a):
    if isinstance(a, np.ndarray):
        return np.asmatrix(a)
    else:
        raise ValueError('Must be of type numpy array')
