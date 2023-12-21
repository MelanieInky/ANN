"""
Let's try and make an MLP from scratch
"""
import numpy as np

class Layer:
    def __init__(self,size,size_last,activation = 'logistic',bias = True):
        #Exclude the bias in size argument
        #The weight are the weight coming TO the layer
        #bias = Include bias as the first element
        self.bias = bias

        self.h = np.zeros(size)
        if(bias):
            self.h[0] = 1 #Bias
            self.w = np.zeros((size-1,size_last))
            self.gradw = np.zeros((size-1,size_last))
        else:
            self.w = np.zeros((size,size_last))
            self.gradw = np.zeros((size,size_last))           
            
        self.delta = np.zeros(size)
        if(activation == 'logistic'):
            self.phi = lambda x : 1 / (1+np.exp(-x))
            self.phi_prime_phi_x = lambda x : x*(1-x)
        elif(activation == 'linear'):
            self.phi = lambda x : x
            self.phi_prime_phi_x = lambda x : 1
        pass
    
    def forward(self,x):
        """Forward the current layer with the input x
        Returns self.h for the next layer
        Args:
            x (_type_): _description_
        """
        if(self.bias):
            self.h[1:] = self.w @ x
            self.h[1:] = self.phi(self.h[1:])
        else:
            self.h = self.w @ x
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
        

class MLP:
    def __init__(self,X,d,n_hlayer,width,act = 'logistic'):
        """Main class for the MLP
        Some choices:
        Hidden layers all have the same number of nodes
        Loss function is separable by sample
        Args:
            X (_type_): Input, without the bias added
            D (_type_): Output, for now only one output per sample allowed
            n_layer (_type_): Number of hidden layers
            width (_type_): Number of nodes in each hidden layer,
            including bias
            act: Type of activation function
        """
        self.n_samples , self.n_input = X.shape
        self.n_output = d.shape[1]
        self.n_hlayer = n_hlayer
        self.h_width = width
        self.X = X
        self.d = d
        self.layer_list = []
        if(n_hlayer > 0): #MLP
            layer = Layer(self.h_width,self.n_input+1,activation=act) #Dont forget bias!
            self.layer_list.append(layer)
            for i in range(n_hlayer-1):
                layer = Layer(self.h_width,self.h_width,activation=act)
                self.layer_list.append(layer)
            out_layer = Layer(self.n_output,self.h_width,'linear',bias = False)
            self.layer_list.append(out_layer)
        else:
            #Perceptron
            raise NotImplementedError('Perceptron not done for now')
        pass
        
    def reset_nodes(self):
        #Reset all nodes except weight ofc
        for layer in self.layer_list:
            layer.h.fill(0)
            layer.h[0] = 1
            layer.gradw.fill(0)
            layer.delta.fill(0)
            

    
    def set_weight(self,value = 1):
        """Set the weights all to 1, for testing
        """
        for layer in self.layer_list:
            layer.w.fill(value)
    
    def set_randomweight(self,std_dev):
        for layer in self.layer_list:
            layer.w = np.random.normal(0,std_dev,np.shape(layer.w))
    
    def forward(self, n):
        # Do a forward pass for the n'th sample
        #First, input to first hidden node
        x = self.X[n]
        x = np.concatenate(([1],x)) #Add the bias
        #Now, do the hidden nodes
        #Do the first layer
        h = self.layer_list[0].forward(x)
        #May or may not work
        for i in range(1,self.n_hlayer+1):
            h = self.layer_list[i].forward(h)
        return h #y(x;w)

    
    
    def backward(self,n):
        #Do a backward propagation and increment the gradients
        #First, do output to last hidden node
        #We assume linear activation for now
        delta = np.zeros(self.n_output)
        out_layer = self.layer_list[-1]
        y_out = out_layer.h
        #To be changed for more complex loss functions
        #Also need to think to support multiclass thing
        delta = (self.d[n] - y_out) / self.n_samples
        out_layer.delta = delta
        if(self.n_hlayer == 0):
            # TODO : verify
            input = self.X[n]
            input = np.concatenate(([1],input))
        else:
            input = self.layer_list[-2].h
        out_layer.gradw -= np.outer(delta,input)
        #Now backpropagate for each layer
        for i in range(self.n_hlayer-1, -1 , -1):
            layer = self.layer_list[i]
            next_layer = self.layer_list[i+1]
            if(i != 0):
                input = self.layer_list[i-1].h
            else:
                input = self.X[n]
                input = np.concatenate((np.array([1]),input))
            #Compute the deltatilde
            delta = layer.backward(next_layer,input) #Something like that    
        
    def learn1(self,n):
        """Learning based a a single sample

        Args:
            n (_type_): _description_
        """
        for i in range(1000):
            self.forward(n)
            self.backward(n)
            for layer in self.layer_list:
                layer.learn()
            
    def learn_all(self,n_epoch = 1000):
        """Batch learning, for a number of epoch

        Args:
            n_epoch (int, optional): _description_. Defaults to 1000.
        """
        for i in range(n_epoch):
            self.reset_nodes()
            for j in range(self.n_samples):
                self.forward(j)
                self.backward(j)
            for layer in self.layer_list:
                layer.learn()
    
    def loss(self):
        pass
    
    def fd_gradient(self,n,i,j,k):
        #For debugging purposes, bruteforce the gradient
        """Returns the difference between the backprop calculated 
        gradient of w_jk in the node i, for the sample n 
        and
        a finite difference approximation of it
        """
        h = 1e-6
        self.reset_nodes()
        self.forward(n)
        self.backward(n)
        grad = self.layer_list[i].gradw[j,k]
        loss = (self.d[n]-self.layer_list[-1].h)**2 / (2*self.n_samples)
        self.reset_nodes()
        self.layer_list[i].w[j,k] += h
        self.reset_nodes()
        self.forward(n)
        self.backward(n)
        loss2 = (self.d[n]-self.layer_list[-1].h)**2 / (2*self.n_samples)
        grad_fd = (loss2-loss)/h
        print(grad_fd-grad)
        
        
        
        
        


X = np.array([[1,2,3],
              [2,1,4],
              [3.5,1,0]])
y = np.array([[1],
              [2],
              [3.5]])

mlp = MLP(X,y,2,4,act='logistic')
mlp.set_randomweight(0.1)
mlp.learn_all(100000)


print(mlp.forward(0))
print(mlp.forward(1))
print(mlp.forward(2))


#It's alive!!!
