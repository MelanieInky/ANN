"""Helpful function for making convolutions operations"""

import numpy as np

DEBUG = False


def convolution(inp: np.ndarray, kernel: np.ndarray, stride=1):
    """Make a convolution according to the kernel

    Args:
        inp (np.ndarray): The inp image, single or multiple channel
        kernel (np.ndarray): _description_
        stride (int, optional): _description_. Defaults to 1.
    """
    # For now we forget about zero padding

    shape_k = kernel.shape
    shape_inp = inp.shape
    if len(shape_k) != len(shape_inp):
        raise ValueError("The dimensions of the inp and kernel must be the same")
    if len(kernel.shape) == 3:
        k_x, k_y, k_z = kernel.shape
        inp_x, inp_y, inp_z = inp.shape
        if k_z != inp_z:
            raise ValueError("The depth of the inp and the kernel must be the same")
    elif len(kernel.shape) == 2:
        k_x, k_y = kernel.shape
        inp_x, inp_y = inp.shape
    else:
        raise ValueError("Input must be 2d or 3d")

    if isinstance(stride, int):
        s_x = stride
        s_y = stride
    elif isinstance(stride, (tuple, list, np.ndarray)):
        if len(stride) != 2:
            raise ValueError("The stride must either be an int or of length 2")
        s_x = stride[0]
        s_y = stride[1]
    out_x = int((inp_x - k_x) / s_x) + 1
    out_y = int((inp_y - k_y) / s_y) + 1

    # Make an array of lists
    inp_out_tbl = np.frompyfunc(list, 0, 1)(np.empty((inp_x, inp_y), dtype=object))

    out = np.zeros((out_x, out_y))
    for i in range(out_x):
        for j in range(out_y):
            sliced = inp[i * s_x: i * s_x + k_x, j * s_y: j * s_y + k_y]
            for m in range(k_x):
                for n in range(k_y):
                    if DEBUG:
                        print(
                            f"inp ({i*s_x+m},{j*s_y+n}), linked to out ({i},{j}), via weight ({m},{n})"
                        )
                    inp_out_tbl[i * s_x + m, j * s_y + n].append((i, j, m, n))
            out[i, j] = np.sum(sliced * kernel)
    return out, inp_out_tbl


def make_correspondence_table(input_dim, kernel_dim, stride):

    k_x, k_y, k_z = kernel_dim
    inp_x, inp_y, inp_z = input_dim
    s_x = stride[0]
    s_y = stride[1]

    out_x = int((inp_x - k_x) / s_x) + 1
    out_y = int((inp_y - k_y) / s_y) + 1

    # Make an array of lists
    inp_out_tbl = np.frompyfunc(list, 0, 1)(np.empty((inp_x, inp_y), dtype=object))
    kernel_tbl = np.frompyfunc(list, 0, 1)(np.empty((k_x, k_y), dtype=object))
    for i in range(out_x):
        for j in range(out_y):
            for m in range(k_x):
                for n in range(k_y):
                    if DEBUG:
                        print(
                            f"inp ({i*s_x+m},{j*s_y+n}), linked to out ({i},{j}), via weight ({m},{n})"
                        )
                    inp_out_tbl[i * s_x + m, j * s_y + n].append((i, j, m, n))
                    # Format: (output_x, output_y, kernel_x,kernel_y)
                    kernel_tbl[m, n].append((i * s_x + m, j * s_y + n, i, j))
                    # Format: (input_x, input_y, output_x,output_y)
    return inp_out_tbl, kernel_tbl


def pooling(inp, kernel_size, type="max"):
    """Make a pooling operation

    Args:
        inp (_type_): _description_
        kernel_size (_type_): _description_
        type (str, optional): _description_. Defaults to 'max'.
    """
    return 0


if __name__ == "__main__":
    a = np.arange(9).reshape((3, 3))
    k1 = np.array([[1, 1], [1, 1]])
    b, in_out = convolution(a, k1)
    # c = convolution(a,k1,stride=2)
    # k2 = np.array([[1]])
    # d = convolution(a,k2,stride = (1,2))
