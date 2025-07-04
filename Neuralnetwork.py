import numpy as np

"""layer_dense class"""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X,self.weights)+self.biases
        return self.output
    def backward(self,dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #  # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
    
    def normalization(self):
        """to implement"""
        pass
        # self.output = self.output/max_value 
        # return self.output


"""output normalization """

"""loss classes"""
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        self.data_loss = np.mean(sample_losses)
       
        return self.data_loss
class loss_mse(Loss):
    """
    """
    def forward(self, y_pred, y_true):
        self.diff = (y_pred - y_true)
        self.numerator = self.diff**2
        return self.numerator
    def backward(self):
        self.dinputs = 2*self.diff/np.size(self.numerator)
        return self.inputs
    
class loss_CategoricalCrossEntropy(Loss):
    """
    """
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:# you have passed a scalar value
            correct_conf = y_clipped[range(samples),y_true]#this extracts rows first and the second part tells u the prediction we wanted

        elif len(y_true.shape) ==2:
            correct_conf = np.sum(y_clipped*y_true, axis = 1)

        negative_likelihoods = -np.log(correct_conf)
        return negative_likelihoods
  
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


"""Activation classes"""
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs #remember input values
        self.output = np.maximum(0,inputs)
    def backward(self,dvalues):
        # Since we need to modify the original variable,
        #  # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        #  # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
  
class Activation_Softmax:
    def forward(self,inputs):
        sum_vals = np.exp(inputs-np.max(inputs))
      
        self.output = sum_vals / np.sum(sum_vals, axis=1,keepdims=True)
    def backward(self,dvalues):
        return self

class Activation_SoftMax_crosscategorical_loss(Loss):
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = loss_CategoricalCrossEntropy()
        return 

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if(y_true.shape) == 2: # if values of y_true are one hot encoded turn them into discrete values of 1d array
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1
        #Normalize gradients
        self.dinputs = self.dinputs/samples
        

        return 
    

"""optimizers"""
class optimizer_SGD:
    def __init__(self, learning_rate = 1):
        self.learning_rate = learning_rate
        return 
    def update(self,layer):
        """
        Updates the layer's weights and biases using calculated gradients.
        Args:
            learning_rate (float): The learning rate for parameter updates.
        """
        # Standard Gradient Descent update rule: parameter = parameter - learning_rate * gradient
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Sequential:
    def __init__(self, layers:list):
        self.weights = None
        self.biases = None
        self.layers  = layers

    
    def compile_net(self, optimizer,loss, metrics:list = None, learning_rate = 1):
        self.loss = loss()
        self.optimizer = optimizer(learning_rate)
        self.metrics = metrics
        return 
    
    def fit(self,input_data, output_data, epochs = 5, batch_size = 1, shuffle_batch = False):
        """
        function takes in input data and output data aswell as an optimizer and metric(optional)

        """
        x_batches,y_batches = self._create_batches(input_data,output_data,)
        for epoch in range(epochs):
            for X,y_true in zip(x_batches,y_batches):
                y_pred = self._forward(X)
                self.loss.calculate(y_pred,y_true)
                self._backward(y_true)

        return 
    
    def evaluate(self, input_data, output_data):
        #TODO
        """#todo"""
        pass

    def predict(self, input_data):
        self.output = self._forward(input_data)
        return self.output
    
    def _forward(self, input_data):
        outdata = None
        for layer in self.layers:
            outdata = layer.forward(input_data)
            input_data = outdata
        return outdata
    
    def  _backward(self,ybatch):
        self.loss.backward(self.loss.output,ybatch)
        prev_layer = self.loss
        for layer in reversed(self.layers):
            layer.backward(prev_layer.dinputs)
            prev_layer = layer
        return 

            

    def _create_batches(self, input_data, output_data, batch_size = 1 ,  shuffle = False):
        """
        Divides the rows of a NumPy array into batches of a specified size.

        Args:
            data (np.ndarray): The input NumPy array (features/rows).
                            Expected shape: (num_samples, num_features).
            batch_size (int): The desired size of each batch.
            shuffle (bool): If True, shuffles the data before batching.
                            (Recommended for training data).

        Yields:
            np.ndarray: A batch of data. The last batch might be smaller.
        """
        num_samples = input_data.shape[0]

        # Optional: Shuffle the data
        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            input_data = input_data[indices]
            input_data = output_data[indices]

        # Iterate through the data in chunks of batch_size
        for i in range(0, num_samples, batch_size):
            # Slice the data to get the current batch
            x_batch,y_batch = input_data[i:i + batch_size],output_data[i:i+batch_size]
            yield x_batch,y_batch



if __name__ == "__main__":
    pass