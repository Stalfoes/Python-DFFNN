# Python-DFFNN

Description:
  A python-made deep feed-forward neural network. Created using Numpy library for matrix calculations. Based off of Michael Nielsen's online book "Neural Networks and Deep Learning".

Notes:
  * This is based off of Michael Nielsen's online book "Neural Networks and Deep Learning".
  * I've based the general algorithm off of his process in his book, but I've used Numpy's matrix capabilities to simplify the calculations for various equations.
  * The network doesn't have any private methods so all methods are able to be publicly called. For intended usage, only use the methods outlined below.
  
Outline of usage:
  1) You can initialize the network using the constructor.
  2) You can start the network's training using the method "learn".
    i) The training data is of a very specific format so the network will have an easy time uncompressing the data.
    ii) The format is:
      tuple(numpy_array, numpy_array)
      Where the numpy arrays represent the inputs and outputs respectively.
  3) You can use the networks specifications to feedforward an input to get an output. This is useful to actual applications. This can be done using the inputOutput method.
    i) This method can take 1 or 2 parameters.
    ii) When 1 parameter is specified (input activations), the returned data are the output activations in the form of a numpy array.
    iii) When 2 parameters are specified (input, desired output), the returned data are the output activations and a boolean representing whether the network successfully got the desired output or not.
  4) You are able to import your own weights and biases into the network using the method setWeightsAndBiases(weights, biases).
