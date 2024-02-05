from Layer import *

class NeuralNetwork():
    def __init__(self,X,Y,loss=None) -> None:
        self.layer_list=[]
        self.X = X
        self.Y = Y
        self.input_dim = X[0].shape
        self.output_dim = Y[0].shape
        self.current_input_dim = self.input_dim
        self.label_dim = self.output_dim #Alias for output_dim
        

        
        
    def add_dense_layer(self,n_nodes,activation='linear'):
        if self.layer_list:
            self.layer_list[-1].is_last_layer = False
        #Check if the input is flat
        if(len(self.current_input_dim) == 1):
            print(activation)
            layer = DenseLayer(n_nodes,self.current_input_dim[0],activation)
            self.current_input_dim = (n_nodes,)
            self.layer_list.append(layer)
        else:
            layer = FlattenLayer(self.current_input_dim)
            self.layer_list.append(layer)
            self.current_input_dim = layer.output_dim
            self.add_dense_layer(n_nodes,activation)
            
    
            
    def forward_nn(self,input):
        for layer in self.layer_list:
            out = layer.forward(input)
            input = out
    
    def backward_nn(self):
        pass
    
    def __str__(self) -> str:
        str = 'Neural network with:\n'
        str += f'Input dimensions {self.input_dim}\n'
        str += f'Output dimensions {self.output_dim}\n'
        str += f'----- Layers: -----\n'
        for layer in self.layer_list:
            str += layer.__str__()
        return str
        
    
            
        
        


if __name__ == '__main__':
    #Test the dense layer thingy
    X = np.zeros((4,3,2)) #4 images of size 3,2
    Y = np.zeros((2,10))
    ann = NeuralNetwork(X,Y)
    ann.add_dense_layer(5,'logistic')
    ann.add_dense_layer(7,'softmax')
    print(ann)