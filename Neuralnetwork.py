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
        self.output= self.diff**2
        return self.output
    def backward(self,dvalues=0, y_true=0):
        self.dinputs = 2*self.diff/np.size(self.output)
        return self.dinputs
    
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
        #negative likelyhood
        self.output = -np.log(correct_conf)
        return self.output
  
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
        return self.output
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
    """learning rate decay  
    alpha = alphaprev/1+decay*t
    where t is the iteration number 
    decay is usually chosen to be 0.001

    momentum to improve will be w(new) = w(previous)-alpha*dl/dw + momentum factor*prevweight updates
    """
    def __init__(self, learning_rate = 1,decay = 0.0, momentum = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        return 
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1+self.decay*self.iterations)
    def update(self,layer):
        """
        Updates the layer's weights and biases using calculated gradients.
        Args:
            learning_rate (float): The learning rate for parameter updates.
        """
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates  = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights

            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            
            layer.bias_updates = bias_updates
        else:
            # Standard Gradient Descent update rule: parameter = parameter - learning_rate * gradient
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates= - self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self):
        self.iterations+=1



class Sequential:
    optimizers = {
        'sgd': optimizer_SGD,
        
    }
    loss_func = {
        'mse': loss_mse,
        'cce': loss_CategoricalCrossEntropy
    }
    def __init__(self, layers:list):
        self.weights = None
        self.biases = None
        self.layers  = layers

    
    def compile_net(self, optimizer,loss, metrics:list = None, learning_rate = 1):
        optimizers = {
        'sgd': optimizer_SGD,
        
        }
        loss_func = {
            'mse': loss_mse,
            'cce': loss_CategoricalCrossEntropy
        }
        self.loss = loss_func[loss]()
       
        self.optimizer = optimizers[optimizer](learning_rate)
        self.metrics = metrics
        return 
    
    def fit(self,input_data, output_data, epochs = 5, batch_size = -1, shuffle_batch = False):
        """
        function takes in input data and output data aswell as an optimizer and metric(optional)

        """
    
        x_batches,y_batches = self._create_batches(input_data,output_data,batch_size=batch_size)
        for epoch in range(epochs):
            self.loss_avg = 0
            for X,y_true in zip(x_batches,y_batches):
                y_pred = self._forward(X)
                batch_loss = self.loss.calculate(y_pred,y_true)
                self.loss_avg+=batch_loss
                # print('loss', l, 'acc:', accuracy)
                
                self._backward(y_true)
                
            print("epoch:" ,epoch, "\nloss:", self.loss_avg/np.size(x_batches))
        return 
    
    def evaluate(self, input_data, output_data):
        #TODO
        """#todo"""
        pass

    def predict(self, input_data):
        self.output = self._forward(input_data)
        return self.output
    
    def get_loss(self, input_data, output_data):
        if hasattr(self, 'output'):
            return self.loss.calculate(self.output,output_data)
        else:
            return 'cannot get loss: \n please try running predict or fit the model'
        

    def _forward(self, input_data):
        outdata = None
        i = 0
        for layer in self.layers:

            outdata = layer.forward(input_data)
            input_data = outdata
            i+=1
    
        return outdata
    
    def  _backward(self,ybatch):
        self.loss.backward(self.loss.output,ybatch)
        self.optimizer.pre_update()
        prev_layer = self.loss
        for layer in reversed(self.layers):
            layer.backward(prev_layer.dinputs)
            prev_layer = layer
            if hasattr(layer,'weights'):
                self.optimizer.update(layer)
        self.optimizer.post_update()
   
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

        if num_samples != output_data.shape[0]:
            raise ValueError("input_data and output_data must have the same number of samples.")

        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            input_data = input_data[indices]
            output_data = output_data[indices] # Corrected this line

        if batch_size == -1:
            batch_size = num_samples

        x_batches = []
        y_batches = []

        for i in range(0, num_samples, batch_size):
            x_batch = input_data[i:i + batch_size]
            y_batch = output_data[i:i + batch_size]
            x_batches.append(x_batch)
            y_batches.append(y_batch)

        return x_batches, y_batches


if __name__ == "__main__":
    pass