from abc import ABC , abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    def get_weights(self):
        if self.w is None:
            return None
        return self.w

######### DENSE LAYER ##############

class DenseLayer(Layer):
    def __init__(self,size,input_size,activation = 'logistic',bias = True):
        
        self.bias = bias
        self.h = np.zeros(size)
        if(bias):
            self.h[0] = 1 #Bias
            self.w = np.zeros((size-1,input_size))
            self.gradw = np.zeros((size-1,input_size))
        else:
            self.w = np.zeros((size,input_size))
            self.gradw = np.zeros((size,input_size))           
            
        self.delta = np.zeros(size)
        if(activation == 'logistic'):
            self.phi = lambda x : 1 / (1+np.exp(-x))
            self.phi_prime_phi_x = lambda x : x*(1-x)
        elif(activation == 'linear'):
            self.phi = lambda x : x
            self.phi_prime_phi_x = lambda x : 1
        pass
    
    def forward(self,input):
        """Forward the current layer with the input input
        Returns self.h for the next layer
        Args:
            x (_type_): _description_
        """
        if(self.bias):
            self.h[1:] = self.w @ input
            self.h[1:] = self.phi(self.h[1:])
        else:
            self.h = self.w @ input
            self.h = self.phi(self.h)     
        return self.h
    
    
    
    def backward(self,next_layer,input):
        """Use after a forward pass to update the gradients values of
        the weights leading to this layer, treating it as the hidden
        layer
        

        # Args:
            next_layer (Layer): The next layer (in the 'next_layer'
            is the output layer sense.
            input (1d array): The input, from a further layer.
        """
        #We need to ditch the first delta 
        #Since its for the bias.
        if(next_layer.bias == True):
            delta = (next_layer.delta[1:] @ next_layer.w)
        else:
            delta = (next_layer.delta @ next_layer.w)  
        #Checkout the bias
        if(self.bias == True):          
            delta[1:] *= self.phi_prime_phi_x(self.h[1:])
        else:
            delta *= self.phi_prime_phi_x(self.h)            
        self.delta = delta
        gradw = np.outer(delta,input)
        if(self.bias == True):
            self.gradw -= gradw[1:] #Dump the first row
        else:
            self.gradw -= gradw
        pass
    
    def learn(self):
        #TODO, set learning rate
        #Finally, learn through the magic of gradient descent
        self.w = self.w - 0.001* self.gradw    
        
