from abc import ABC , abstractmethod
import numpy as np
from Convolution import convolution
from Activation import *

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
    
    def reset_all(self):
        #Reset all the output and gradients
        self.out.fill(0)
        self.grad_bw.fill(0)
        self.grad_b.fill(0)
        

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
        

class ConvolutionLayer(Layer):
    def __init__(self,input_dim,
                 kernel_dim,
                 stride = 1,
                 n_filters = 1,activation='logistic',bias=True):
        
        #Setting the dimensions variables
        if(len(input_dim) != len(kernel_dim)):
            raise ValueError('The dimensions of the input and kernel must be the same')
        if(len(kernel_dim) == 3): 
            self.k_x , self.k_y , self.k_z = kernel_dim
            self.i_x , self.i_y , self.i_z = input_dim
            if(self.k_z != self.i_z):
                raise ValueError('The depth of the input and the kernel must be the same')
        elif(len(kernel_dim) == 2):
            self.k_x , self.k_y  = kernel_dim
            self.k_z = 1
            self.i_z = 1
            self.i_x , self.i_y  = input_dim
        else:
            raise ValueError('Input must be 2d or 3d')
        
        
        #Setting the stride
        if(isinstance(stride,int)):
            self.s_x = stride
            self.s_y = stride
        elif(isinstance(stride,(tuple,list,np.ndarray))):
            if(len(stride) != 2):
                raise ValueError('The stride must either be an int or of length 2')
            self.s_x = stride[0]
            self.s_y = stride[1]
        
        self.stride = (self.s_x,self.s_y)
        #Preset the output dimensions
        out_x = int((self.i_x-self.k_x)/self.s_x) + 1
        out_y = int((self.i_y-self.k_y)/self.s_y) + 1
        self.out = np.zeros((n_filters,out_x,out_y))
        
        #Set the bias vectors
        self.b = np.zeros(n_filters)
        self.grad_b = np.zeros_like(self.b)
        #Kernel weights, all in one big 4d table,
        #First dim is the filter number
        self.w = np.zeros((n_filters,self.k_x,self.k_y,self.k_z))
        self.grad_w = np.zeros_like(self.w)
        self.n_filters = n_filters
        
        #Set the activation object
        if(activation=='logistic'):
            self.activation = Logistic()
        elif(activation=='linear'):
            self.activation = Linear()
        
        
    def forward(self,input):
        for f in range(self.n_filters):
            kernel = self.w[f]
            self.out[f], _ = convolution(input,kernel,self.stride)
            self.out[f] = self.activation.phi(self.out[f])


if __name__ == '__main__':
    conv_layer = ConvolutionLayer((3,3),(2,2),activation='linear')
    A = np.arange(9).reshape((3,3,1))
    conv_layer.w[0] = np.array([[1,1],[1,1]]).reshape((2,2,1))
    conv_layer.forward(A)
    conv_layer.out
    
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
