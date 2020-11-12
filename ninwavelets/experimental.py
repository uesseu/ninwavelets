from ninwavelets import WaveletBase, CWTMode
from ninwavelets.base import WaveletFormula
from ninwavelets.wavelets import MorletFormula
from logging import getLogger, INFO, basicConfig, NullHandler, Logger
from typing import Union, List, Iterator, Callable, Tuple, cast, Optional, Dict, Any
import numpy as np
try:
    import cupy as cp
    import cupyx.scipy.fftpack as cx_fft
    cp.fft.ifft(cp.arange(1, 10, 1))
except ImportError as error:
    print(error)
    print('Cupy could not be loaded.')
    cp = np

logger = getLogger('ninwavelets')
logger.addHandler(NullHandler())
Array = Union[np.ndarray, cp.ndarray]
Numbers = Union[List[float], np.ndarray, range]

class FACWT(WaveletBase):
    """
    Experimental cwt method.
    Fast approximated continuous wavelet transform.
    It seems to be slow because of python code.
    But may have potential to be extremely fast...if it is written in C lang.
    """
    def app_cwt(self, wave: Array, freqs: Numbers,
                 reuse_wavelets: bool = True,
                 band_rate: float = 0.01,
                 logger: Logger = logger) -> Array:
        '''cwt
        Run fast approximated CWT.

        wave: np.ndarray
            Wave to analyze
        freqs: Union[List[float], range, np.ndarray]
            Frequencies. It can be argument of cwt, but it is slow.
            If you want to calculate repeatedly, you should run
            make_fft_wavelets before cwt, and freqs should be None.
        max_freq: int
            Max Frequency
        reuse_wavelets: bool
            Reuse wavelets or not.
        logger: Logger
            logger

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        dimension: int = len(wave.shape)
        wave_shape: tuple = wave.shape
        remake_plan = False
        if self.cuda:
            wave = wave.astype(cp.complex)
        # This if statement can not be method, for performance.
        if (not reuse_wavelets) or (self.fft_wavelets is None):
            if self.cuda:
                remake_plan = True
            sid = ''.join((str(wave_shape), str(id(freqs))))
            if sid in self._kept['fft'].keys():
                self.fft_wavelets = self._kept['fft'][sid]
            else:
                self.make_fft_wavelets(freqs, wave_shape[-1] / self.sfreq)
                self.bands = self.fft_wavelets > band_rate
                self.fft_wavelets = [w[w > band_rate] for w in self.fft_wavelets]
                if((not self.cache_limit) or
                   self.cache_limit > len(self._kept['fft'])):
                    self._kept['fft'].update({sid: self.fft_wavelets})
        ncp = cp if self.cuda else np
        fft_wave = ncp.fft.fft(wave)
        ii = np.zeros((wave.shape[0], freqs.shape[0], wave.shape[-1]), np.complex)
        def get_res(i: int) -> None:
            r = self.fft_wavelets[i] * fft_wave[:, self.bands[i]]
            start = int(freqs[i] - r.shape[-1] / 2)
            stop = int(start + r.shape[-1])
            ii[:, i, start: stop] = r[:]
        [get_res(i) for i in range(freqs.shape[-1])]
        return ncp.fft.ifft(ii)


class MorseFormula(WaveletFormula):
    def __init__(self, r: float, b: float) -> None:
        self.r: float = r
        self.b: float = b

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

class Morse(FACWT):
    '''
    Morse Wavelets.
    It is new wavelet, which is orthonormal.
    Unlike Morlet Wavelets, it is robust for any parameters.
    Originally, Generalized Morse wavelet is
    Frourier transformed wave.

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
    cache_limit: Optional[int]
        Number of wavelets, which will be kept in the instance.
        If it is None, it will keep all the wavelets.

    Returns
    -------
    As constructor, Morse instance its self.
    '''
    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 real_wave_length: float = 1., cuda: bool = False,
                 cache_limit: Optional[int] = 10) -> None:
        super(Morse, self).__init__(sfreq, real_wave_length, cuda,
                                    cache_limit=cache_limit)
        self.formula = MorseFormula(r, b)
        self.mode = CWTMode.Fast
