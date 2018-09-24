import random
import numpy as np
import math

class Network(object):
    def __init__(self, sizes):
        """ The constructor for the Network.
        This handles the network's initialization.
        Parameters:
            sizes: an array where the layer sizes are listed.
        """
        # Now we will create the biases and weights according to the sizes
        # Store the sizes
        self.sizes = sizes
        # Store the length of the network
        self.L = len(sizes)
        # Create gaussian-distributed weights and biases according to the specified sizes
        self.weightsMatrix = [np.random.randn(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]
        self.biasesMatrix = []
        for s in range(1, len(sizes)):
            self.biasesMatrix.append(np.random.randn(sizes[s],1))
        # Reformat the biasesMatrix to arrays because we can't use 4x1
        # arrays or else the feedforward method breaks
        for i in range(len(self.biasesMatrix)):
            self.biasesMatrix[i] = toArray(self.biasesMatrix[i])
            
            
    def setWeightsAndBiases(self, weights, biases):
        """ Sets the weights and biases to the specified input.
        Used to re-create a previous network.
        Parameters:
            weights: The specified weights.
            biases: The specified biases.
        Returns:
            None
        """
        self.weightsMatrix = weights[:]
        self.biasesMatrix = biases[:]
       
    
    def refreshEpoch(self):
        """ This method handles all the errors numpy has with type-changing.
        While testing, the types of the matrices and arrays would change
        from epoch-to-epoch and were uncontrollable. This method attempts to
        fix this issue by deleting the variables and then re-initializing
        them.
        Also used by the inputOutput method.
        Parameters:
            None
        Returns:
            None
        """
        # We try to delete the variables.
        try:
            del self.errors_over_epoch[:]
            del self.activations_over_epoch[:]
            del self.activationsMatrix[:]
            del self.Zs[:]
            del self.ErrorsMatrix[:]
            del self.epoch_summary[:]
            del self.num_correct_in_epoch
        except:
            # Not a problem
            pass
        # Re-initialize the variables with no values.
        self.errors_over_epoch = []
        self.activations_over_epoch = []
        self.activationsMatrix = []
        self.Zs = []
        self.ErrorsMatrix = []
        self.epoch_summary = []
        # Epoch summary should be a list of tuples (size 4)
        # List:
        #  Tuple:
        #    Vector - Input
        #    Vector - Expected output
        #    Vector - Generated output
        #    Boolean - True or false whether we got it right or not
        self.num_correct_in_epoch = 0        
    
    
    def learn(self, num_epochs, training_data, eta, output_type):
        """ The main method for training the network.
        Stochastic gradient descent is not used, as the network is not particularly
        intended to handle large amounts of training data.
        See the readme to see the specified formatting of the training_data input.
        Parameters:
            num_epochs: The number of epochs to train the network on.
                        This is the number of times to run through the training_data.
            training_data: The given inputs and outputs to train with respectively.
                           See the readme for important formatting.
            eta: The learning rate. This scales the 'weight' of each training example.
            output_type: A string-type representing the amount of output you desire into
                         the console from the network.
                Valid inputs are:
                    'all', 'last', & 'none'
        """
        # We train the network for num_epochs
        for epoch in range(num_epochs):
            # Refresh the epoch and the variables
            self.refreshEpoch()
            # Shuffle the training data
            random.shuffle(training_data)
            # For each training example in training_data...
            for training_data_index in range(len(training_data)):
                # The batch pair is the tuple of inputs and outputs
                batch_pair = training_data[training_data_index]
                # Set the first activations to the input
                self.activationsMatrix = [batch_pair[0]]
                # Set the Zs to zeros based on the sizes
                self.Zs = []
                self.Zs = [np.zeros(s) for s in self.sizes[1:]]
                
                # Feedforward algorithm
                self.feedforward()
                
                # Compute the error for the last layer, L
                self.ErrorsMatrix = [np.zeros(s) for s in self.sizes[1:]]
                # Get the desired outputs from the batch_pair
                Y = batch_pair[1]
                # This next part should fix our typing issue
                # Convert each element in Zs to a numpy array
                for i in range(len(self.Zs)):
                    self.Zs[i] = np.array(self.Zs[i])
                
                # Calculate the cost derivative
                costDerivative = self.costDerivative(self.activationsMatrix[-1], Y)
                # Set the last errors, based on given equations
                self.ErrorsMatrix[-1] = np.multiply(costDerivative, sigmoid_prime(self.Zs[-1]))
                
                #Backpropogation algorithm
                self.backpropogate()
                
                # Save the errors and activations for this training set
                # We need these later for evaluation at the end of the epoch
                self.errors_over_epoch.append(self.ErrorsMatrix)
                self.activations_over_epoch.append(self.activationsMatrix)
                # Ask if the network got the example correct
                self.evaluateSingle(batch_pair)
                
            # Now we want to use gradient descent to reduce the cost
            self.gradientDescent(eta, training_data, epoch)
                
            # Now we should evaluate the network
            self.evaluateEpoch(epoch, len(training_data), output_type, num_epochs)            
    
    
    def costDerivative(self, A, Y):
        """ The mean-squared cost derivative.
        Parameters:
            A: The input activations
            Y: The desired output activations
        Returns:
            None
        """
        return A - Y
    
    
    def evaluateSingle(self, batch_pair):
        """ Evaluate the single training example
        We generate the self.epoch_summary and record whether we got the example
        problem correct or not.
        Parameters:
            batch_pair: The pair of inputs and outputs.
        Returns:
            None
        """
        # The last activations of the network
        last_activations = self.activationsMatrix[-1]
        # Concatenate the last layer's activations onto the tuple
        batch_triple = batch_pair + tuple([last_activations])
        # The desired activations
        theoretical_activations = batch_pair[1]
        # We make sure the type and shape of the last_activations are correct
        if last_activations.shape[0] == 1:
            last_activations = toArray(last_activations)   
        # The index of the max value in the theoretical activations
        max_index_theoretical = np.argmax(theoretical_activations)
        # The index of the max value in the experimental activations
        max_index_experimental = np.argmax(last_activations)
        # Boolean value of whether the two indeces match
        bool_maxes_match = max_index_theoretical == max_index_experimental
        # We concatenate the boolean onto the the return
        batch_quadruple = batch_triple + tuple([bool_maxes_match])
        # We increase the number of examples we got correct in the epoch
        # if we got the example correct
        if bool_maxes_match:
            self.num_correct_in_epoch += 1
        # Append the information onto the summary variable
        self.epoch_summary.append(batch_quadruple)
    
    
    def gradientDescent(self, eta, training_data, epoch):
        """ The gradient descent method.
        Here we take the cost function and minimize the cost using
        the value of the negative gradient.
        Parameters:
            eta
            training_data
            epoch: The index of the epoch that the network is currently evaluating.
        Returns:
            None
        """
        # The constant mutliplier of the equations
        multiplier = -eta / len(training_data)
        # For each layer, except the last one...
        for i in range(0, self.L - 1):
            # Get the weights before the change of the layer
            BeforeWeights = self.weightsMatrix[i]
            # Initialize the summation variable
            Summation = 0
            # For each training example in the training_data...
            for x in range(len(training_data)):
                # Get the errors and format them to an Nx1 array
                errors = to1xNMatrix(self.errors_over_epoch[x][i]).transpose()
                # Get the activations and format them into a 1xN array
                activations = to1xNMatrix(self.activations_over_epoch[x][i])
                # Matrix multiply the activations and errors and add them to the sum
                Summation += np.matmul(errors, activations)
            # The change in the biases is the sum * the multiplier
            ChangeInWeights = multiplier * Summation
            # Set the weights after the change
            AfterWeights = BeforeWeights + ChangeInWeights
            # Set the weights to the newly changed weights
            self.weightsMatrix[i] = AfterWeights
            
            # Get the biases before the change of the layer
            BeforeBiases = self.biasesMatrix[i]
            # Reset the summation variable
            Summation = 0
            # For each training example in the training_data...
            for x in range(len(training_data)):
                # The sum gets increased by the errors
                Summation += self.errors_over_epoch[x][i]
            # The change is multiplier * sum
            ChangeInBiases = multiplier * Summation
            # Set the biases after to the change applied to the biases before
            AfterBiases = BeforeBiases + ChangeInBiases
            # Set the biasese to the newly changed biases
            self.biasesMatrix[i] = toArray(AfterBiases)        
    
    
    def backpropogate(self):
        """ The backpropogation algorithm.
        We generate the errors matrix so we can perform gradient descent.
        Paramters:
            None
        Returns:
            None
        """
        # We backpropogate from the second last layer to the second layer
        for i in range(self.L-2, 2, -1):
            # Split up the equation into two parts because it is long
            # The left side is the transpose of the weights matrix of the previous layer 
            # multiplied with the errors matrix of the previous layer
            left_side = np.matmul(self.weightsMatrix[i + 1].transpose(), self.ErrorsMatrix[i + 1])
            # The final product, we set to the errors matrix of the current layer
            # We multiply the left by the derivative of the sigmoid function
            self.ErrorsMatrix[i] = np.multiply(left_side, sigmoid_prime(self.Zs[i]))
        
        
    def feedforward(self):
        """ The feedforward algorithm method.
        We feedforward the activations from the first layer to the last.
        Parameters:
            None
        Returns:
            None
        """
        # Starting at the first layer and moving toward the last layer...
        for i in range(0, self.L - 1):
            # Calculate the Zs based off of the current weights, biases and activations
            self.Zs[i] = np.matmul(self.weightsMatrix[i], self.activationsMatrix[i]) + self.biasesMatrix[i]
            # Set the next activation from the sigmoid function applied to the calculated Z
            nextActivation = sigmoid(self.Zs[i])
            # We reformat the variable to the desired specifications
            if nextActivation.shape[0] == 1:
                nextActivation = nextActivation.transpose()
                nextActivation = toArray(nextActivation)
            # Add the calculated activation to the next spot in the activations matrix
            self.activationsMatrix.append(nextActivation)
            
            
    def evaluateEpoch(self, epoch, num_trained, output_type, n_epochs):
        """ Evaluate the epoch of the network.
        We do the printing into the console for the network.
        Parameters:
            epoch: The index of the epoch we're on
            num_trained: the total number of examples we trained with
            output_type: The output amount
            n_epochs
        Returns:
            None
        """
        # Calculate the performance of the network
        performance = self.num_correct_in_epoch / num_trained * 100.0
        # Depending on the output type, print the performance of the network
        if output_type == 'all' or (output_type == 'last' and epoch == n_epochs-1):
            print('Epoch {0}: {1:.5g}% => {2} / {3}'.format(epoch+1, performance, self.num_correct_in_epoch, num_trained))
        
        
    def endprinting(self, array1, array2):
        """ This formats two arrays into the console nicely.
        Parameters:
            array1: The first array
            array2: The second array
        Returns:
            None
        """
        for n in range(len(array1)):
            print('{0:.4g}   {1:.4g}'.format(array1[n], array2[n]))
        print('\n')
        
        
    def inputOutput(self, *io):
        """ The standalone function for creating output by feed-forwarding
        given input.
        See the readme for further documentation on this function.
        Parameters:
            i) Combination:
                input_activations: The first layer's input activations to be
                                   feed-forwarded
           ii) Combination:
                input_activations: The first layer's input activations to be
                                   feed-forwarded
                output_activations: The last layer's activations to be evaluated
                                    against
        Returns:
            i) Combination:
                last_activations: The fed-forward activations from the input
           ii) Combination:
                last_activations: The fed-forward activations from the input
                correct: A boolean value representing whether we got the example
                         correct or not
        """
        if len(io) == 1:
            # We've only recieve input activations
            # We just need to return the output activations
            # Get the activations
            input_activations = io[0]
            # Refresh the variables so we don't interfere with them
            self.refreshEpoch()
            # Set the initial activations to the input
            self.activationsMatrix = [input_activations]
            # Initialize the Zs variable to zeros according to each layer
            self.Zs = []
            self.Zs = [np.zeros(s) for s in self.sizes[1:]]    
            # Feed-forward algorithm
            self.feedforward()
            # Return the result of the feed-forward
            return self.activationsMatrix[-1]
        elif len(io) == 2:
            # We recieved the inputs and expected outputs
            # We should return the output activations and whether it got it right or wrong
            # Get the inputs and expected outputs
            input_activations = io[0]
            expected_output_activations = io[1]
            # Refresh the variables so we don't interfere with them
            self.refreshEpoch()
            # Set the initial activations to the input
            self.activationsMatrix = [input_activations]
            # Initialize the Zs variable to zeros according to each layer
            self.Zs = []
            self.Zs = [np.zeros(s) for s in self.sizes[1:]]    
            # Feed-forward algorithm
            self.feedforward()
            # Evaluate the output from the feed-forward algorithm
            self.evaluateSingle(tuple([input_activations, expected_output_activations]))
            if self.num_correct_in_epoch == 1:
                # If we got the example correct...
                return (self.activationsMatrix[-1], True)
            else:
                # If we got the example incorrect
                return (self.activationsMatrix[-1], False)                
        else:
            raise ValueError('Can only recieve maximum 2 input parameters')
        

"""These two functions will output arrays if arrays are inputted
They are used for calculations involving arrays.
"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    #return ReLU(z)
    
    
def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))
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
    return ret


"""Now conversion functions to and from arrays and matrices of 1 dimension. This is to bug-fix."""
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
