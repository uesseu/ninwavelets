# -*- coding: utf-8 -*-
"""Orthogonal wavelet module.
This is an orthogonal wavelet module in ninwavelets.
This is not perfect.

    from matplotlib import pyplot as plt
    fig: plt.Figure = plt.figure()
    ax = fig.add_subplot(111)
    ord = 63
    cs = get_daubechies_coeff(4)
    length: int = 1024
    wavelet = DaubechiesWavelet(cs).make_all(length, ord)
    ax.plot(wavelet.father)
    ax.plot(wavelet.mother)
    ax.set_title('DaubechiesWavelet')
    plt.show()
    wave = np.sin(np.arange(0, 1 << 13, 1) / 200)
    j = daubechies_mra(wave, 8)
    fig = plt.figure()
    pls = 8
    axes = [fig.add_subplot((pls+1) * 100 + 11 + n)
            for n in range(pls+1)]
    for n in range(pls):
        axes[n].plot(np.arange(0, len(wave), 2**(n+1)), j[n])
    fig.suptitle('MRA')
    plt.show()
    x = np.sin(np.arange(1024)/100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x)
    ax.set_title('Sin wave')
    plt.show()
    res_haar, m = haar_mra(x)
    y = haar_imra(res_haar, m)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title('Inverted MRA')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x - y)
    ax.set_title('Error')
    plt.show()
"""

import numpy as np
from functools import reduce
from operator import mul
from typing import List, Optional, Union, Tuple, cast
from scipy.sparse import csc_matrix, csr_matrix


daubechies_coeff: List[List[float]] = [
    [1, 1],
    [0.6830127, 1.1830127, 0.3169873, -0.1830127],
    [0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175],
    [0.32580343, 1.01094572, 0.89220014, -0.03957503, -
        0.26450717, 0.0436163, 0.0465036, -0.01498699],
    [0.22641898, 0.85394354, 1.02432694, 0.19576696, -0.34265671, -
        0.04560113, 0.10970265, -0.00882680, -0.01779187, 4.71742793 * 1e-3],
    [0.15774243, 0.69950381, 1.06226376, 0.44583132, -0.31998660, -0.18351806,
     0.13788809, 0.03892321, -0.04466375, 7.83251152 * 1e-4, 6.75606236 * 1e-3,
     -1.52353381 * 1e-3],
    [0.11009943, 0.56079128, 1.03114849, 0.66437248, -0.20351382, -0.31683501,
     0.1008467, 0.11400345, -0.05378245, -0.02343994, 0.01774979,
     6.07514995 * 1e-4, -2.54790472 * 1e-3, 5.00226853 * 1e-4],
    [0.07695562, 0.44246725, 0.95548615, 0.82781653, -0.02238574, -0.40165863,
     6.68194092 * 1e-4, 0.18207636, -0.02456390, - 0.06235021, 0.01977216,
     0.01236884, -6.88771926 * 1e-3, -5.54004549 * 1e-4, 9.55229711 * 1e-4,
     -1.66137261 * 1e-4],
    [0.05385035, 0.34483430, 0.85534906, 0.92954571, 0.18836955, -0.41475176,
     -0.13695355, 0.21006834, 0.043452675, -0.09564726, 3.54892813 * 1e-4,
     0.03162417, -6.67962023 * 1e-3, -6.05496058 * 1e-3, 2.61296728 * 1e-3,
     3.25814671 * 1e-4, -3.56329759 * 1e-4, 5.5645514 * 1e-5],
    [0.03771716, 0.26612218, 0.74557507, 0.97362811, 0.39763774, -0.35333620,
     -0.27710988, 0.18012745, 0.13160299, -0.10096657, -0.04165925, 0.04696981,
     5.10043697 * 1e-3, -0.01517900, 1.97332536 * 1e-3, 2.81768659 * 1e-3,
     -9.69947840 * 1e-4, -1.64709006 * 1e-4, 1.32354367 * 1e-4,
     -1.875841 * 1e-5]]


def get_daubechies_coeff(n: int) -> np.ndarray:
    """
    Get Daubechies coefficient numbers as a numpy array.
    """
    return np.array(daubechies_coeff[(n >> 1) - 1])


class DaubechiesWavelet:
    """
    A class to make orthogonal wavelet.
    Now, it only yields Daubechies wavelet.

    >>> cs = np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127])
    >>> wavelet = OrthogonalWavelet(cs)
    >>>
    """
    def __init__(self, coeff: List[float]) -> None:
        self.coeff: Optional[np.ndarray] = np.array(coeff)
        self.mother_coeff: Optional[np.ndarray] = coeff_father2mother(coeff)
        # Flow 1
        self.order: Optional[int] = None
        # Flow 2
        self.fft_father: Optional[np.ndarray] = None
        # Flow 3
        self.father: Optional[np.ndarray] = None
        # Flow 4
        self.mother: Optional[np.ndarray] = None
        # Flow 5
        self.fft_mother: Optional[np.ndarray] = None
        self.flow: int = 0

    def _make_length(self, length: int):
        return (len(self.coeff) - 1) * length

    def make_all(self, length: int, order: int = 63,
                 mod_num: int = 100) -> 'DaubechiesWavelet':
        """
        Make father wavelet and mother wavelet.
        It is based on FFT, but includes modification of errors.
        FFT and modification may be faster method, I think.

        length: int
            Length of the wavelet.
            It should be multiply of (power of two) and
            (Number of coefficiency - 1).
            If it is not such number, it is useless.
        order: int
            It is a number to make the father.
            Bigger is good, but because of modification, is not important.
            It crashed when order is bigger than 63 in my computer.
        mod_num: int = 100
            Modification number. It is based on iteration.
        """
        self.make_fourier_father(length, order)
        self.make_father(length)
        self.modify_father(mod_num)
        self.make_mother()
        self.make_fft_mother()
        return self

    def make_fourier_father(self, length: int, order: int = 63) -> np.ndarray:
        """
        Make fourier transformed scaling function.
        So called father wavelet.

        length: int
            Length of the wavelet.
            It should be multiply of (power of two) and
            (Number of coefficiency - 1).
            If it is not such number, it is useless.
        order: int
            It is a number to make the father.
            Bigger is good, but because of modification, is not important.
            It crashed when order is bigger than 63 in my computer.
        """
        self.order = order
        self.fft_father = make_fourier_father(self.coeff,
                                              self._make_length(length),
                                              order)
        return self.fft_father

    def make_father(self, length: int) -> np.ndarray:
        """
        Make scaling function from fourier transformed wavelet.
        It is so called father wavelet.
        """
        self.father = np.fft.ifft(self.fft_father).real
        self.father *= self._make_length(length) / np.sqrt(1 << self.order)
        self.father = self.father - self.father[-1]
        return self.father

    def make_mother(self, to_mother: bool = True) -> np.ndarray:
        """
        Make mother wavelet.
        """
        self.mother = father2mother(self.father, self.coeff, to_mother)
        return self.mother

    def make_fft_mother(self) -> np.ndarray:
        """
        Fourier transform mother wavelet.
        """
        self.fft_mother = np.fft.ifft(self.mother)
        return self.fft_mother

    def modify_father(self, n: int) -> np.ndarray:
        """
        Modify rough father wavelet.
        The father wavelet by self.make_father returns only rough version.
        It should be run to make precise version.

        n: int
            Number of iteration.
            If it is big, the wavelet will be precise.
        """
        for i in range(n):
            tmp = self.father - father2mother(self.father, self.coeff, False)
            self.father -= tmp / 2
        self.father -= cast(np.ndarray, self.father)[0]
        self.make_mother()


def _p(k: np.array, cs: list) -> np.ndarray:
    """
    Generates Fourier version of Daubechies wavelet.
    But rough, lowlevel version.
    """
    return sum(cs[n]*np.exp(-n*1j*k) for n in range(len(cs))) / np.sqrt(2)


def make_fourier_father(cs: Union[list, np.ndarray],
                        length: int,  order: int = 64) -> np.ndarray:
    """
    Generates Fourier version of Daubechies wavelet.
    But rough version. After generating rough father wavelet,
    iterating method will be applied.

    cs: Union[List[int], np.ndarray[int]]
        The coefficient values of father wavelet.
    order: int
        The precision of wavelet.
        It should be big if you want precise wavelet.
        In case of my computer, order bigger than 63 crashed.
    """
    ord = np.arange(length) * np.pi / (len(cs) - 1)
    return reduce(mul, (_p(ord/(2**i), cs) for i in range(order)))


def father2mother(father: np.ndarray, cs: np.ndarray,
                  to_mother: bool = True) -> np.ndarray:
    """
    father: np.ndarray
        The father wavelet.
        In this version, father wavelet must be multiple of order - 1.
    order: int
        Order of the father wavelet.
    """
    half_wave = father[0::2]
    half_wave = half_wave - half_wave[-1]
    small_length = int(len(half_wave) / (len(cs)-1))
    times = (n*small_length for n in range((len(cs))))
    gs = coeff_father2mother(cs) if to_mother else cs
    result = np.zeros_like(father)
    for n, start in enumerate(times):
        result[start: start + half_wave.shape[0]] += gs[n] * half_wave[:]
    return result


def coeff_father2mother(cs: np.ndarray) -> np.ndarray:
    """
    Trans form coefficient numbers of orthogonal wavelet
    from father wavelet to mother wavelet.
    """
    gs: np.ndarray = np.zeros_like(cs)
    gs[0::2], gs[1::2] = -cs[0::2], cs[1::2]
    return np.array(list(reversed(gs)))


def mra(wave: np.ndarray, cs: np.ndarray) -> List[np.ndarray]:
    """
    Perform MRA by Daubechies wavelet.
    Before using this, you should make matrix_flow of Daubechies.
    It can be made by make_daubechies_matrix_flow
    """
    gs = coeff_father2mother(cs)
    ds: List[np.ndarray] = []
    wv: np.dnarray
    cp: np.dnarray
    while wave.shape[0] > 1:
        wv, cp = [sum(np.roll(wave * coeff, -num)
                      for num, coeff in enumerate(coeffs))
                  for coeffs in (cs, gs)]
        wave = wv[..., 0::2]
        ds.append(cp[..., 0::2])
    return ds

def daubechies_mra(wave: np.ndarray, coeff: int) -> np.ndarray:
    return mra(wave, get_daubechies_coeff(coeff))

def haar_mra(x: np.ndarray) -> Tuple[list, float]:
    """
    Haar wavelet transform based on lifting scheme.
    Length of the array must be power of 2.
    """
    mra = []
    while(len(x) > 1):
        mra.append(x[0::2] - x[1::2])
        x = x[1::2] + mra[-1] / 2
    return mra, x.mean()


def haar_imra(mra: list, mean: float = 0) -> np.ndarray:
    """
    Haar wavelet inverse transform based on lifting scheme.
    Length of the array must be power of 2.
    """
    c = np.zeros(1)
    for d in reversed(mra):
        x = np.zeros(len(d)*2)
        x[1::2] = c - d / 2
        x[0::2] = d + x[1::2]
        c = x
    c = c - c.mean() + mean
    return c


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    fig: plt.Figure = plt.figure()
    ax = fig.add_subplot(111)
    ord = 63
    cs = get_daubechies_coeff(4)
    length: int = 1024
    wavelet = DaubechiesWavelet(cs).make_all(length, ord)
    ax.plot(wavelet.father)
    ax.plot(wavelet.mother)
    ax.set_title('DaubechiesWavelet')
    plt.show()
    wave = np.sin(np.arange(0, 1 << 13, 1) / 200)
    j = daubechies_mra(wave, 8)
    fig = plt.figure()
    pls = 8
    axes = [fig.add_subplot((pls+1) * 100 + 11 + n)
            for n in range(pls+1)]
    for n in range(pls):
        axes[n].plot(np.arange(0, len(wave), 2**(n+1)), j[n])
    fig.suptitle('MRA')
    plt.show()
    x = np.sin(np.arange(1024)/100)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x)
    ax.set_title('Sin wave')
    plt.show()
    res_haar, m = haar_mra(x)
    y = haar_imra(res_haar, m)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title('Inverted MRA')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x - y)
    ax.set_title('Error')
    plt.show()
