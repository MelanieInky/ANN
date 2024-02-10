import unittest
from Convolution import convolution , make_correspondence_table
import numpy as np
import scipy as sc


class TestConvolutionMethods(unittest.TestCase):
    def setUp(self):
        self.test_inp = np.arange(20).reshape(5, 4)
        self.test_ker = np.array([[1, 2], [3, 4]])
        self.test_ker2 = np.array([[1, 2, 3], [4, 5, 6]])
        self.arr_1 = np.arange(9).reshape(3, 3, 1)
        self.arr_2 = np.arange(8, -1, -1).reshape((3, 3, 1))

        self.test_inp3d = np.concatenate((self.arr_1, self.arr_2), 2)
        ker1 = np.array([[1, 2], [3, 4]])
        ker2 = np.array([[5, 6], [7, 8]])
        self.test_ker3d = np.concatenate((ker1.reshape(2, 2, 1), ker2.reshape(2, 2, 1)), 2)

    def test_convolution(self):
        expected_out = np.array([[34, 44, 54], [74, 84, 94], [114, 124, 134], [154, 164, 174]])
        conv = convolution(self.test_inp, self.test_ker, 1)
        np.testing.assert_array_equal(conv, expected_out)
        # Testing the stride method
        conv = convolution(self.test_inp, self.test_ker, stride=(2, 1))
        expected_out = np.array([[34, 44, 54], [114, 124, 134]])
        np.testing.assert_array_equal(conv, expected_out)
        # Testing the stride again
        conv = convolution(self.test_inp, self.test_ker, stride=2)
        expected_out = np.array([[34, 54], [114, 134]])
        np.testing.assert_array_equal(conv, expected_out)

        # Testing different kernel shape
        expected_out = np.array([[85, 106], [169, 190], [253, 274], [337, 358]])
        conv = convolution(self.test_inp, self.test_ker2, stride=1)
        np.testing.assert_array_equal(conv, expected_out)

        # And some testing in 3d
        conv = convolution(self.test_inp3d, self.test_ker3d)
        expected_out = np.array([[176, 160], [128, 112]])
        np.testing.assert_array_equal(conv, expected_out)

        conv = convolution(self.test_inp3d, self.test_ker3d, (2, 1))
        expected_out = np.array([[176, 160]])
        np.testing.assert_array_equal(conv, expected_out)

    def test_make_correspondence_table(self):
        inp_out_tbl , kernel_tbl = make_correspondence_table((2,2),(2,1),1)
        exp_inp_out_tbl = [[[(0,0,0,0)],[(0,1,0,0)]],
                            [[(0,0,1,0)],[(0,1,1,0)]]]
        self.assertCountEqual(inp_out_tbl, exp_inp_out_tbl)
