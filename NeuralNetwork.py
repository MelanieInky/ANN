from Layer import *

class NeuralNetwork():
    def __init__(self,X,Y,loss=None) -> None:
        self.layer_list=[]
        self.X = X
        self.Y = Y
        self.current_input_dim = X[0].shape
        self.label_dim = self.Y[0].shape
        
    def add_dense_layer(self,n_nodes):
        #Check if the input is flat
        if(len(self.current_input_dim) == 1):
            layer = DenseLayer(n_nodes,self.current_input_dim[0])
            self.current_input_dim = (n_nodes,)
            self.layer_list.append(layer)
        else:
            layer = FlattenLayer(self.current_input_dim)
            self.layer_list.append(layer)
            self.current_input_dim = layer.output_dim
            self.add_dense_layer(n_nodes)
            
    
            
    def forward_nn(self,input):
        for layer in self.layer_list:
            out = layer.forward(input)
            input = out
    
    
            
        
        


if __name__ == '__main__':
    #Test the dense layer thingy
    X = np.zeros((4,3,2)) #4 images of size 3,2
    Y = np.zeros(10)
    ann = NeuralNetwork(X,Y)
    ann.add_dense_layer(5)
    ann.add_dense_layer(7)
    print(ann.layer_list[0])