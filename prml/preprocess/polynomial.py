import itertools
import functools
import numpy as np


class PolynomialFeature(object):
    """
    polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features

        Parameters
        ----------
        degree : int
            degree of polynomial
        """
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        """
        transforms input array with polynomial features

        Parameters
        ----------
        x : (sample_size, n) ndarray
            input array

        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            polynomial features
        """
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            # https://docs.python.org/zh-cn/3/library/itertools.html
            ## combinations_with_replacement 例子:
            ### combinations_with_replacement('ABCD', 2)
            ### AA AB AC AD BB BC BD CC CD DD
            for items in itertools.combinations_with_replacement(x_t, degree):
                # reduce 高级程序设计课堂测试有实现过这个函数:
                # https://docs.python.org/zh-cn/3/library/functools.html#functools.reduce
                ## 这里就是累乘.
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
