"""Helpful function for making convolutions operations"""
import numpy as np

def convolution(input,kernel,stride=1,zero_padding=False):
    """Make a convolution according to the kernel

    Args:
        inp (_type_): _description_
        kernel (_type_): _description_
        stride (int, optional): _description_. Defaults to 1.
        zero_padding (bool, optional): Whether to zero pad or not. Returns the same dimensions if we have zp
    """
    #For now we forget about zero padding
    
    shape_k = kernel.shape
    shape_inp = input.shape
    if(len(shape_k) != len(shape_inp)):
        raise ValueError('The dimensions of the input and kernel must be the same')
    if(len(kernel.shape) == 3): 
        k_x , k_y , k_z = kernel.shape
        inp_x , inp_y , inp_z = input.shape
        if(k_z != inp_z):
            raise ValueError('The depth of the input and the kernel must be the same')
    elif(len(kernel.shape) == 2):
        k_x , k_y  = kernel.shape
        inp_x , inp_y  = input.shape
    else:
        raise ValueError('Input must be 2d or 3d')
    
    
    if(isinstance(stride,int)):
        s_x = stride
        s_y = stride
    elif(isinstance(stride,(tuple,list,np.ndarray))):
        if(len(stride) != 2):
            raise ValueError('The stride must either be an int or of length 2')
        s_x = stride[0]
        s_y = stride[1]
        print(s_x)
        print(s_y)
    out_x = int((inp_x-k_x)/s_x) + 1
    out_y = int((inp_y-k_y)/s_y) + 1

    out = np.zeros((out_x,out_y))
    for i in range(out_x):
        for j in range(out_y):
            slice = input[i*s_x:i*s_x+k_x,j*s_y:j*s_y+k_y]
            out[i,j] = np.sum(slice * kernel)
    return out



    
    
def pooling(input,kernel_size,type='max'):
    """Make a pooling operation

    Args:
        input (_type_): _description_
        kernel_size (_type_): _description_
        type (str, optional): _description_. Defaults to 'max'.
    """
    return 0


if __name__ == '__main__':
    a = np.arange(9).reshape((3,3))
    k1 = np.array([[1,1],[1,1]])
    b = convolution(a,k1)
    c = convolution(a,k1,stride=2)
    k2 = np.array([[1]])
    d = convolution(a,k2,stride = (1,2))
    
    