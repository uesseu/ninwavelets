import numpy as np
from multiprocessing import Value, Process
from typing import Callable, List, Tuple
import ctypes
from functools import reduce, wraps
from operator import mul
from time import time


def np_parallel(cores: int = 2, c_type: type = ctypes.c_float,
                complex: bool = False) -> Callable:
    """
    A decorator to make a function written in numpy run in parallel.
    The argument should be a np.ndarray,
    and result should be np.ndarray.
    In case of function which returns complex,
    complex option must be True.

    Arguments
    ----------
    cores: int = 2
        Job number to use.
    c_type: type = ctypes.c_float
        c_type to use. It is just for shared memory.
    complex: bool = False
        If the result is complex, set it True.
        Then, it returns result with imaginary part.

    Returns
    ----------
    """
    def wrap(func: Callable) -> Callable:
        resolver = NumpyParallel(func, cores, c_type, complex)
        @wraps(func)
        def wrap2(*array: np.ndarray) -> np.ndarray:
            return resolver.run(array)
        return wrap2
    return wrap
        

def _make_share(array: np.ndarray, c_type: type) -> Value:
    return Value(reduce(mul, reversed(array.shape + (c_type,))))
def _divide(array: np.ndarray, cores: int) -> List[np.ndarray]:
    return [ar[:] for ar in np.array_split(array, cores)]
def _get_obj(value: Value) -> np.ndarray:
    return np.ctypeslib.as_array(value.get_obj())
class NumpyParallel:
    def __init__(self, func: Callable, cores: int = 2,
                 c_type: type = ctypes.c_float, complex: bool = False):
        self.func = func
        self.cores = cores
        self.c_type = c_type
        self.complex = complex
        self.arrays: List[List[np.ndarray]]  # divided

    def _div_func(self, arrays: List[np.ndarray], n: int) -> None:
        result = self.func(*arrays)
        shapes = [ar.shape[0] for ar in self.arrays[0]]
        start = sum([shape for shape in shapes[:n]])
        stop = sum([shape for shape in shapes[:n+1]])
        if self.complex:
            _get_obj(self.share_c[0])[start: stop] = result.real[:]
            _get_obj(self.share_c[1])[start: stop] = result.imag[:]
        else:
            _get_obj(self.share)[start: stop] = result[:]

    def run(self, arrays: Tuple[np.ndarray]) -> np.ndarray:
        if self.complex:
            self.share_c = [_make_share(arrays[0], self.c_type)
                            for n in range(2)]
        else:
            self.share = _make_share(arrays, self.c_type)
        self.arrays = [_divide(ar, self.cores) for ar in arrays]
        args = list(zip(*self.arrays))
        ps = [Process(target=self._div_func, args=(args[n], n))
              for n in range(self.cores)]
        tuple(p.start() for p in ps)
        tuple(p.join() for p in ps)
        if self.complex:
            result = np.empty(arrays[0].shape, dtype=np.complex)
            result.real, result.imag = [self.share_c[n][:] for n in range(2)]
            return result
        return np.array(self.share)


if __name__ == '__main__':
    @np_parallel(complex=True, cores=6)
    def test(arg: np.ndarray) -> np.ndarray:
        return np.fft.fft(arg)
    def test_single(arg: np.ndarray) -> np.ndarray:
        return np.fft.fft(arg)

    @np_parallel(complex=True, cores=6)
    def test2(arg: np.ndarray, arg2: np.ndarray) -> np.ndarray:
        return arg**arg2
    def test2_single(arg: np.ndarray, arg2: np.ndarray) -> np.ndarray:
        return arg**arg2
    arg = np.array([np.arange(0.1, 1, 0.001) for n in range(10000)])
    arg2 = np.array([np.arange(0.1, 1, 0.001) for n in range(10000)])


    t = time()
    test(arg)
    print(time() - t)

    t = time()
    test_single(arg)
    print(time() - t)

    t = time()
    test2(arg, arg2)
    print(time() - t)

    t = time()
    test2_single(arg, arg2)
    print(time() - t)
    # print(np.count_nonzero(test2(arg) - test(arg) > 0.0001))
