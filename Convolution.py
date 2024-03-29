"""Helpful function for making convolutions operations"""

import numpy as np

DEBUG = False


def convolution(inp: np.ndarray, kernel: np.ndarray, stride=1, zp=False):
    """Make a convolution(in the ANN sense) according to the kernel
    With stride 1 and 2d input, this is equivalent to
    sc.signal.correlate2d(inp, kernel, mode='valid')

    Args:
        zp: bool: whether to use zero padding or not, not implemented yet
        inp (np.ndarray): The inp image, single or multiple channel
        kernel (np.ndarray): _description_
        stride (int, optional): _description_. Defaults to 1.
    Returns:
        out: np.ndarray: The output of the convolution.
    """
    # For now, we forget about zero padding
    if zp:
        raise NotImplementedError("Zero padding is not implemented")
    shape_k = kernel.shape
    shape_inp = inp.shape
    if len(shape_k) != len(shape_inp):
        raise ValueError("The dimensions of the inp and kernel must be the same")
    if len(shape_k) == 3:
        k_x, k_y, k_z = shape_k
        inp_x, inp_y, inp_z = shape_inp
        if k_z != inp_z:
            raise ValueError("The depth of the inp and the kernel must be the same")
    elif len(shape_k) == 2:
        k_x, k_y = shape_k
        inp_x, inp_y = shape_inp
    else:
        raise ValueError("Input must be 2d or 3d")
    if k_x > inp_x or k_y > inp_y:
        raise ValueError("Kernel size must be less or equal to the input size")
    s_x, s_y = get_stride_xy(stride)
    out_x = int((inp_x - k_x) / s_x) + 1
    out_y = int((inp_y - k_y) / s_y) + 1

    out = np.zeros((out_x, out_y))
    for i in range(out_x):
        for j in range(out_y):
            sliced = inp[i * s_x: i * s_x + k_x, j * s_y: j * s_y + k_y]
            out[i, j] = np.sum(sliced * kernel)
    return out


def make_correspondence_table(input_dim, kernel_dim, stride):
    k_x, k_y = kernel_dim[:2]
    inp_x, inp_y = input_dim[:2]
    s_x, s_y = get_stride_xy(stride)

    out_x = int((inp_x - k_x) / s_x) + 1
    out_y = int((inp_y - k_y) / s_y) + 1

    # Make an array of lists
    inp_out_tbl = [[[] for _ in range(inp_y)] for _ in range(inp_x)]
    kernel_tbl = [[[] for _ in range(k_y)] for _ in range(k_x)]

    for i in range(out_x):
        for j in range(out_y):
            for m in range(k_x):
                for n in range(k_y):
                    if DEBUG:
                        print(
                            f"inp ({i * s_x + m},{j * s_y + n}), linked to out ({i},{j}), via weight ({m},{n})"
                        )
                    inp_out_tbl[i * s_x + m][j * s_y + n].append((i, j, m, n))
                    # Format: (output_x, output_y, kernel_x,kernel_y)
                    kernel_tbl[m][n].append((i * s_x + m, j * s_y + n, i, j))
                    # Format: (input_x, input_y, output_x,output_y)
    return inp_out_tbl, kernel_tbl


def get_stride_xy(stride):
    if isinstance(stride, int):
        s_x = stride
        s_y = stride
    elif isinstance(stride, (tuple, list, np.ndarray)):
        if len(stride) != 2:
            raise ValueError("The stride must either be an int or of length 2")
        s_x = stride[0]
        s_y = stride[1]
    else:
        raise ValueError("Unknown stride type, accepted format is either int or tuple,list,nd.array")
    return s_x, s_y


def pooling(inp, kernel_size, type="max"):
    """Make a pooling operation

    Args:
        inp (_type_): _description_
        kernel_size (_type_): _description_
        type (str, optional): _description_. Defaults to 'max'.
    """
    return 0


if __name__ == "__main__":
    DEBUG = True
    a = np.arange(9).reshape((3, 3))
    k1 = np.array([[1, 1], [1, 1]])
    b = convolution(a, k1)

    inp_out_tbl, kernel_tbl = make_correspondence_table((2, 2), (2, 1), 1)
    print(inp_out_tbl)
