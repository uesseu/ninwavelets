from .base import WaveletBase, CWTMode
from typing import Union, List, cast, Optional
import numpy as np
try:
    import cupy as cp
except ImportError as error:
    print(error)
    print('Cupy could not be loaded.')


class Morse(WaveletBase):
    '''
    Morse Wavelets.
    It is new wavelet, which is orthonormal.
    Unlike Morlet Wavelets, it is robust for any parameters.

    Example.
    >>> morse = Morse(1000, r=3., b=17.5)
    >>> freq = 60
    >>> time = np.arange(0, 0.3, 0.001)
    >>> sin = np.array(np.sin(time * freq * 2 * np.pi))
    >>> result = morse.power(sin, np.arange(1, 100))
    >>> plt.imshow(result, cmap='RdBu_r')
    >>> plt.gca().invert_yaxis()
    >>> plt.title('CWT of 60Hz sin wave')
    >>> plt.show()

    Parameters
    ----------
    sfreq: float
        Sampling frequency. This behaves like sfreq of mne-python.
    b: float
        beta value
    r: float
        gamma value. 3 may be good value.
    real_wave_length: float
        Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
    cuda: bool
        Use cuda
    keep_num: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, Morse instance its self.
    '''

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 real_wave_length: float = 1., cuda: bool = False,
                 keep_num: Optional[int] = None) -> None:
        super(Morse, self).__init__(sfreq, real_wave_length, cuda,
                                    keep_num=keep_num)
        self.r: float = r
        self.b: float = b
        self.mode = CWTMode.Fast
        self.help = '''This is inverse Fourier transformed MorseWavelet.
Originally, Generalized Morse wavelet is
Frourier transformed wave.
'''

    def cp_trans_formula(self, freqs: cp.ndarray,
                         freq: float = 1.) -> cp.ndarray:
        np_freqs = cp.asnumpy(freqs)
        step = cp.asarray(np.heaviside(np_freqs, np_freqs))
        freqs = freqs / freq
        wave = 2. * (step * cp.power(freqs, self.b) *
                     cp.exp((self.b / self.r) *
                            (1.
                             - cp.power(freqs, self.r))
                            ))
        return wave

    def trans_formula(self, freqs: np.ndarray, freq: float = 1.) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
        '''
        freqs = freqs / freq
        step = np.heaviside(freqs, freqs)
        wave = 2. * (step * np.float_power(freqs, self.b)
                     * np.exp((self.b / self.r)
                              * (1. - np.float_power(freqs, self.r))))
        return wave


class Morlet(WaveletBase):
    '''
    Morlet Wavelets.
    A traditional analystic wavelet, which is used widely.
    When sigma value is too small, the wavelet waveform is distorted.
    It is like Gabor Wavelet, which is not orthonormal.

    Example.
    >>> morlet = Morlet(1000, sigma=7.)
    >>> freq = 60
    >>> time = np.arange(0, 0.3, 0.001)
    >>> sin = np.array(np.sin(time * freq * 2 * np.pi))
    >>> result = morlet.power(sin, np.arange(1, 100))
    >>> plt.imshow(result, cmap='RdBu_r')
    >>> plt.gca().invert_yaxis()
    >>> plt.title('CWT of 60Hz sin wave')
    >>> plt.show()

    Parameters
    ----------
    sfreq: float
        Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float
        sigma value
    gabor: bool
        Use Gabor Wavelet.
    read_wave_length: float
        Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
        Peak wave * length is the length of wavelet.
    cuda: bool
        Use cuda.
    keep_num: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, Morlet instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 real_wave_length: float = 1.,
                 gabor: bool = False, cuda: bool = False,
                 keep_num: Optional[int] = None) -> None:
        super(Morlet, self).__init__(sfreq, real_wave_length, cuda,
                                     keep_num=keep_num)
        self.mode = CWTMode.Fast
        self.sigma = sigma
        self.c = np.float_power(1
                                + np.exp(-np.square(self.sigma))
                                - 2 * np.exp(-3 / 4 * np.square(self.sigma)),
                                -1/2)
        self.k = 0 if gabor else np.exp(-np.float_power(self.sigma, 2) / 2)

    def cp_trans_formula(self, freqs: cp.ndarray,
                         freq: float = 1.) -> cp.ndarray:
        peak_freq = self.sigma / (1. - cp.exp(-self.sigma * freq))
        freqs = freqs / freq * peak_freq
        result = (self.c * cp.pi ** (-1/4) *
                  (cp.exp(-cp.square(self.sigma-freqs) / 2) -
                   self.k * cp.exp(-cp.square(freqs) / 2)))
        return result

    def trans_formula(self, freqs: np.ndarray, freq: float = 1) -> np.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        return (self.c * np.float_power(np.pi, -1/4)
                * (np.exp(-np.square(self.sigma-freqs) / 2)
                   - self.k * np.exp(-np.square(freqs) / 2)))

    def cp_formula(self, timeline: cp.ndarray, freq: float = 1) -> np.ndarray:
        return (self.c * (cp.pi ** (-1 / 4))
                * cp.exp(-cp.square(timeline) / 2)
                * (cp.exp(self.sigma * 1j * timeline) - self.k))

    def formula(self, timeline: np.ndarray, freq: float = 1) -> np.ndarray:
        return (self.c * np.float_power(np.pi, (-1 / 4))
                * np.exp(-np.square(timeline) / 2)
                * (np.exp(self.sigma * 1j * timeline) - self.k))

    def peak_freq(self, freq: float) -> float:
        return self.sigma / (1. - np.exp(-self.sigma * freq))



class MexicanHat(WaveletBase):
    '''
    MexicanHat Wavelets.
    It is wavelet of real number.
    And so, you can see rainbow like result after cwt.

    Parameters
    ----------
    sfreq: float
        Sampling frequency.  This behaves like sfreq of mne-python.
    sigma: float
        sigma value
    read_wave_length: float
        Length of wavelet.
    cuda: bool
        Use cuda
    keep_num: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, MexicanHat instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7,
                 real_wave_length: float = 1., cuda: bool = False,
                 keep_num: Optional[int] = None) -> None:
        super(MexicanHat, self).__init__(sfreq, real_wave_length, cuda,
                                         keep_num=keep_num)
        self.sigma: float = sigma
        self.mode = CWTMode.Fast
        self.help = ''
        self.cuda = cuda

    def formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        return ((1 - np.power(tc / self.sigma, 2))
                * np.exp(-np.square(tc) / np.square(self.sigma) / 2))

    def cp_formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        return ((1 - cp.power(tc / self.sigma, 2))
                * cp.exp(-cp.square(tc) / cp.square(self.sigma) / 2))

    def peak_freq(self, freq: float) -> float:
        return cast(float, np.sqrt(6.) / (np.pi ** 2))


class Shannon(WaveletBase):
    '''
    Shannon Wavelets.
    When you fourier transform this, you can see rectangle wave.

    Parameters
    ----------
    sfreq: float
        Sampling frequency.  This behaves like sfreq of mne-python.
    read_wave_length: float
        Length of wavelet.
    cuda: bool
        Use cuda
    keep_num: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, Shannon instance its self.
    '''

    def __init__(self, sfreq: float = 1000, 
                 real_wave_length: float = 1., cuda: bool = False,
                 keep_num: Optional[int] = None) -> None:
        super(Shannon, self).__init__(sfreq, real_wave_length, cuda,
                                      keep_num=keep_num)
        self.mode = CWTMode.Fast
        self.help = ''

    def trans_formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        return np.where(tc <= 1., 1., 0)


class Haar(WaveletBase):
    '''
    Haar Wavelets.
    It looks like rectangle wave.
    It is an acient wavelet, and still useful for compressing data.
    But it is not used for analystic way generally.

    Parameters
    ----------
    sfreq: float
        Sampling frequency.
    read_wave_length: float
        Length of wavelet.
    cuda: bool
        Use cuda
    keep_num: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, Shannon instance its self.
    '''
    def __init__(self, sfreq: float = 1000,
                 real_wave_length: float = 1., cuda: bool = False,
                 keep_num: Optional[int] = None) -> None:
        super(Haar, self).__init__(sfreq, real_wave_length, cuda,
                                   keep_num=keep_num)
        self.mode = CWTMode.Convolve
        self.cuda = cuda

    def formula(self, timeline: np.ndarray, freq: float = 1) -> np.ndarray:
        timeline = np.where(np.abs(timeline) >= 1 / freq, timeline, 0)
        timeline = np.where(timeline > 0., 1, timeline)
        timeline = np.where(timeline < 0., -1, timeline)
        return timeline - np.flip(timeline)
