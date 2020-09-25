import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Iterator, Callable, Tuple, cast, Optional, Dict
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial, reduce
from operator import mul
from logging import getLogger, INFO, basicConfig, NullHandler, Logger
import gc

logger = getLogger('ninwavelets')
logger.addHandler(NullHandler())

try:
    import cupy as cp
    import cupyx.scipy.fftpack as cx_fft
except ImportError as error:
    print(error)
    print('Cupy could not be loaded.')

Numbers = Union[List[float], np.ndarray, range]
Array = Union[np.ndarray, cp.ndarray]
Float = Optional[float]
Floats = Optional[Tuple[float, float]]
NORM_CONSTANT = np.sqrt(0.5)



def cp_alloc(array: np.ndarray) -> np.ndarray:
  buf = np.frombuffer(cp.cuda.alloc_pinned_memory(array.nbytes),
                      array.dtype,
                      array.size).reshape(array.shape)
  buf[...] = array
  return buf


def np2cp(*npdata: np.ndarray) -> cp.ndarray:
    '''
    A simple loader from numpy to cupy.
    This function loads data ansyncloneously.
    But if you want to load over 10000, it is slow.
    Testing speed of this function before use it is recommended.
    >>> first = np.arange(1, 1000, 1)
    >>> second = np.arange(1, 1000, 1)
    >>> set_of_data = [first, second]
    >>> cp_first, cp_second = np2cp(*set_of_data)
    '''
    copy_npdata = (cp_alloc(npd) for npd in npdata)
    cupy_mem = (cp.ndarray(npd.shape, npd.dtype) for npd in npdata)
    streams = (cp.cuda.Stream(non_blocking=True) for npd in npdata)
    tuple(cmem.set(cnpd, stream) for cmem, cnpd, stream in zip(cupy_mem, copy_npdata, streams))
    tuple(stream.synchronize() for stream in streams)
    return cupy_mem



def baseline_of(wave: Array, sfreq: float, start: float, stop: float) -> Array:
    return wave[int(start * sfreq): int(stop * sfreq)]


class Baseline:
    '''Baseline
    Class for baseline correction.

    wave: Union[np.ndarray, cp.nparray]
        Wave you want to process.
        It needs to be single wave.
    sfreq: float
        sfreq
    start: float
        Start time of baseline
    stop: float
        Stop time of baseline

    There are these methods.

    - mean: subtraction
    - ratio: division
    - percent: division after subtraction
    - log: log after division
    - zscore: standize after subtraction
    - zlog: log after zscore
    '''
    def __init__(self, wave: Array, sfreq: float,
                 start: float, stop: float, dim: int = 1) -> None:
        self.wave = wave
        shape = reduce(mul, wave.shape)
        self.baseline = wave.reshape(shape)[int(start * sfreq):
                                            int(stop * sfreq)]
        self.basemean = self.baseline.mean(axis=-1)

    def mean(self) -> Array:
        return self.wave - self.basemean

    def ratio(self) -> Array:
        return self.wave / self.basemean

    def percent(self) -> Array:
        return self.mean() / self.basemean

    def log(self) -> Array:
        return np.log(self.ratio())

    def log10(self) -> Array:
        return np.log10(self.ratio())

    def zscore(self) -> Array:
        return self.mean() / np.std(self.baseline)

    def zlog(self) -> Array:
        return self.log() / np.std(self.baseline)


class SizeError(BaseException):
    def __init__(self, err: str) -> None: print(err)


def pad_to(wave_from: Array, wave_to: Array,
           cuda: bool = False) -> Array:
    from_size, to_size = wave_from.shape[0], wave_to.shape[0]
    if from_size > to_size:
        start = wave_from.shape[0] // 2 - wave_to.shape[0] // 2
        end = wave_from.shape[0] // 2 + wave_to.shape[0] // 2
        return wave_from[start:end]
    else:
        side1 = (to_size - from_size) // 2
        side2 = to_size - from_size - side1
        ncp = cp if cuda else np
        return ncp.pad(wave_from, [side1, side2], 'constant')


def normalize(wave: Array, length: float,
              cuda: bool = False) -> Array:
    ''' Normalize norm of complex array

    Parameters
    ----------
    wave: np.ndarray[np.complex128, ndim=1]
        Wave to normalize.

    Returns
    -------
    np.ndarray[np.complex128, ndim=1]: Normalized wave.
    '''
    return wave * length / np.linalg.norm(wave)


class CWTMode(Enum):
    '''
    Modes of Wavelets.
    These are used as Wavelet.mode

    Normal = 0
        ifft(Convolve(fft(wavelet) @ fft(wave)))
    # Use Wavelet formula only
    # From wavelet formula, make FFTed formula, then convolve.
    # It seems to be normal, but not best way if there is an
    # FFTed formula.

    Fast = 1
        ifft(Convolve(ffted_wavelet @ fft(wave)))
    # Use FFTed formula only.
    # It may be best, if there is FFTed formula.

    Convolve = 2
        Convolve(wavelet @ wave)
    # Just convolve. Slow.

    Reverse = 3
        Convolve(ifft(ffted_wavelet) @ wave)
    # Even if FFTed formula is there, use IFFTed Wavelet, and FFT.
    # This is ugly and not accurate. Just for test code.
    '''
    Normal = 0
    # Use Wavelet formula only
    # From wavelet formula, make FFTed formula, then convolve.
    # It seems to be normal, but not best way if there is an
    # FFTed formula.

    # ifft(Convolve(fft(wavelet) @ fft(wave)))
    Fast = 1
    # Use FFTed formula only.
    # It may be best, if there is FFTed formula.

    # ifft(Convolve(ffted_wavelet @ fft(wave)))
    Convolve = 2
    # From FFTed formula, compute the raw wavelet.
    # Then convolve. Slow.

    # Convolve(wavelet @ wave)
    Reverse = 3
    # Even if FFTed formula is there, use IFFTed Wavelet, and FFT.
    # This is ugly and not accurate. Just for test code.

    # Convolve(ifft(ffted_wavelet) @ wave)

class WaveletFormula:
    """
    Something like an ABC of 'WaveletGenerator' class.
    But, ABC needs to inheriting functions.
    There is some wavelets which has no normal formula.
    And so, it is not written as ABC.
    """
    def peak_freq(self, freq: float) -> float:
        return 1.

    def formula(self, timeline: Array, freq: Union[Array, float]) -> Array:
        ''' formula
        The formula of Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        timeline: np.ndarray[np.float, ndim=1]
            Time value of formula.
        freq: float
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet.
            timeline: np.ndarray:

        freq: float:
        '''
        return timeline

    def cp_formula(self, timeline: Array, freq: Union[Array, float]) -> Array:
        ''' formula
        The formula of Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        timeline: np.ndarray[np.float, ndim=1]
            Time value of formula.
        freq: float
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet.
            timeline: np.ndarray:

        freq: float:
        '''
        return timeline

    def trans_formula(self, freqs: Iterator[float],
                      freq: Union[Array, float] = 1.) -> Array:
        ''' trans_formula
        The formula of Fourier Transformed Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        freqs: np.ndarray[np.float, ndim=1]
            Frequencies.
            If length of time is same as freqs, It is easy to write.
        freq: float
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet: np.ndarray:
        '''
        return freqs

    def cp_trans_formula(self, freqs: Iterator[float],
                         freq: Union[Array, float] = 1.) -> Array:
        ''' trans_formula
        The formula of Fourier Transformed Wavelet.
        Other procedures are performed by other methods.
        This is method with cupy.

        Parameters
        ----------
        freqs: np.ndarray[np.float, ndim=1]
            Frequencies.
            If length of time is same as freqs, It is easy to write.
        freq: float
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet: np.ndarray:
        '''
        return freqs


class WaveletGenerator(WaveletFormula):
    """
    Generator of wavelets.
    It is used as base class of 'WaveletBase'.
    """
    def __init__(self, sfreq: float = 1000, real_wave_length: float = 1.,
                 cuda: bool = False, cache_limit: Optional[int] = 10) -> None:
        '''
        Parameters
        ----------
        sfreq: float
            Sampling frequency.
        real_wave_length: float
            Length of wavelet. When this class run cwt,
            this will be automatically changed.
        cuda: bool
            Whether use cuda or not
        cache_limit: Optional[int]
            Number of wavelets, which will be kept in the instance.
            If it is None, it will keep all the wavelets.
        '''
        self.mode: CWTMode = CWTMode.Fast
        self.sfreq: float = sfreq
        self.help: str = ''
        self.wave_length: int = int(real_wave_length * sfreq)
        self.freq_dist: float
        self.cuda: bool = cuda
        self._freqs: Numbers = None
        self.freqs: Optional[Array] = None
        self.kept: Dict[str, Dict[str, Array]] = {'fft': {}, 'wavelet': {}}
        self.cache_limit = cache_limit
        self.current_fft_wavelets: Optional[Array] = None
        self.fft_wavelets: Optional[Array] = None
        self.wavelets: Optional[Array] = None

    def _setup_trans_shape(self, freq: float, wave_length: int) -> Array:
        '''
        Setup wave shape.
        real_length is length of wavelet(for example, sec or msec)
        wave_length is length of array to analyze.

        Parameters
        ----------
        freq: float
            Base Frequency. For example, 1.
            It must be base frequency.
            You cannot use this for every freqs.
        real_wave_length: float
            Length of wave(sec).

        Returns
        -------
        np.ndarray
            Timeline to calculate wavelet.
        '''
        ncp = cp if self.cuda else np
        result = ncp.arange(0, wave_length, 1) / freq
        return result

    def _setup_waveletshape(self, freq: float, real_length: float = 1,
                            zero_mean: bool = False) -> Array:
        '''
        Setup wave shape.

        Parameters
        ----------
        freq: float
            Base Frequency. For example, 1.
            It must be base frequency.
            You cannot use this for every freqs.
        zero_mean: bool
            Let the center of wave time zero.

        Returns
        -------
        Tuple[float, float]: (one, total)
        '''
        times = (-self.wave_length / 2, self.wave_length / 2, 1) if zero_mean\
            else (0., self.wave_length, 1)
        result = np.arange(*times)\
            * (2 * freq * np.pi / (self.peak_freq(freq) * self.sfreq))
        return cp.asarray(result, np.float64) if self.cuda else result

    def make_fft_wavelet(self, freq: float, real_length: float = 1.) -> Array:
        ''' Make single FFTed wavelet.

        Parameters
        ----------
        freq: float
            Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        ncp = cp if self.cuda else np
        if freq == 0:
            raise ZeroDivisionError
        formula = self.cp_trans_formula if self.cuda else self.trans_formula
        if self.mode in [CWTMode.Fast]:
            t = self._setup_trans_shape(real_length, self.wave_length)
            result = formula(t, freq)
            return result / self.get_wavelet_norm(ncp.fft.ifft(result), (0,))
        else:
            wavelet = self.make_wavelet(freq)
            half = int((self.wave_length - wavelet.shape[0]) / 2)
            wavelet = ncp.hstack((ncp.zeros(half), wavelet, ncp.zeros(half)))
            result = ncp.fft.fft(wavelet)
            result.imag = ncp.abs(result.imag)
            result.real = ncp.abs(result.real)
            return result

    def make_fft_wavelets(self, freqs: Array, real_length: float = 1.) -> Array:
        ''' Make single FFTed wavelet.

        Parameters
        ----------
        freq: float
            Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        ncp = cp if self.cuda else np
        if freqs[0] == 0:
            raise ZeroDivisionError
        formula = self.cp_trans_formula if self.cuda else self.trans_formula
        if self.mode in [CWTMode.Fast]:
            # Make timeline
            t = ncp.tile(
                self._setup_trans_shape(real_length, self.wave_length),
                (freqs.shape[0], 1))
            # Make fft wavelets
            many_freqs = ncp.tile(freqs, (t.shape[1], 1)).T
            result = formula(t, many_freqs)
            # Adjust norm
            divs = self.get_wavelet_norm(ncp.fft.ifft(result), (1,))
            result /= ncp.tile(divs, (result.shape[1], 1)).T
            self.fft_wavelets = result
            return result
        else:
            logger.info('Making ffted wavelet.')
            self.freq_dist = freqs[1] - freqs[0]
            make_w = partial(self.make_fft_wavelet, real_length=real_length)
            self.fft_wavelets = ncp.array(tuple(map(make_w, freqs)))
            return self.fft_wavelets

    def get_wavelet_norm(self, wavelet: Array,
                         axis: Optional[tuple] = None) -> Array:
        '''
        Get norm of wavelet.

        Parameters
        ----------
        wavelet: Union[np.ndarray, cp.ndarray]
            Wavelet
        '''
        norm = cp.linalg.norm if self.cuda else np.linalg.norm
        result =  NORM_CONSTANT * norm(wavelet, axis=axis)
        return result


    def make_wavelet(self, freq: float) -> Array:
        '''
        Make single wavelet and return.

        Parameters
        ----------
        freq: float

        Returns
        ----------
        Wavelet: Union[cp.ndarray, np.ndarray]
        '''
        ncp = cp if self.cuda else np
        if freq == 0:
            raise ZeroDivisionError
        if self.mode in [CWTMode.Reverse, CWTMode.Fast]:
            timeline = self._setup_trans_shape(freq, self.wave_length)
            if self.cuda:
                wavelet = cp.fft.ifft(self.cp_trans_formula(timeline))
            else:
                wavelet = np.fft.ifft(self.trans_formula(timeline))
            half = int(wavelet.shape[0])
            start, stop = half // 2, half // 2 * 3
            wavelet = ncp.hstack((ncp.conj(ncp.flip(wavelet, 0)),
                                        wavelet))[start: stop]
            wavelet /= self.get_wavelet_norm(wavelet)
        else:
            timeline = self._setup_waveletshape(freq, 1, zero_mean=True)
            formula = self.cp_formula if self.cuda else self.formula
            wavelet = formula(timeline, freq)
            wavelet /= self.get_wavelet_norm(wavelet)
        return wavelet


    def make_wavelets(self, freqs: Numbers) -> Array:
        '''
        Make wavelets. It returnes list of wavelet.

        Parameters
        ----------
        freqs: List[float]
            Frequencies.

        Returns
        -------
        MorseWavelet: np.ndarray
        '''
        logger.info('Making wavelet.')
        ncp = cp if self.cuda else np
        if freqs[0] == 0:
            raise ZeroDivisionError
        if self.mode in [CWTMode.Reverse, CWTMode.Fast]:
            timelines = ncp.array(tuple(self._setup_trans_shape(
                freq, self.wave_length)
                              for freq in freqs))
            if self.cuda:
                wavelet = cp.fft.ifft(self.cp_trans_formula(timelines))
            else:
                wavelet = np.fft.ifft(self.trans_formula(timelines))
            half = int(wavelet.shape[0])
            start, stop = half // 2, half // 2 * 3
            wavelet = ncp.hstack(
                (ncp.conj(ncp.flip(wavelet, 0)), wavelet))[start: stop]
            norms = ncp.array(self.get_wavelet_norm(wavelet, (1,)))
            wavelet /= ncp.tile(norms, (wavelet.shape[1], 1)).T
        else:
            timeline = ncp.array(
                [self._setup_waveletshape(freq, 1, zero_mean=True)
                 for freq in freqs])
            formula = self.cp_formula if self.cuda else self.formula
            wavelet = formula(timeline, freqs)
            divs = ncp.array(self.get_wavelet_norm(wavelet, (1,)))
            wavelet /= ncp.tile(divs, (wavelet.shape[1], 1)).T
        self.wavelets = wavelet
        return wavelet

class WaveletConvolver(WaveletGenerator):
    """
    CWT class which performs cwt by convolve.
    It is very slow.
    """
    def cwt_convolve(self, wave: Array, freqs: Numbers,
                     reuse_wavelets: bool, logger: Logger = logger) -> Array:
        '''
        Backend of cwt in convolve mode. This is not optimized yet.
        Some obsessive people hates DFT and it may be needed.
        And ofcource, I understand their opinions.
        But, because it is hobby, for me, it does not take priority.

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.
        reuse_wavelets: bool
            Reuse wavelets or not.
        logger: Logger
            logger

        Returns
        ----------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        ncp = cp if self.cuda else np
        if (not reuse_wavelets) or (self.wavelets is not None):
            sid = ''.join((str(wave.shape), str(id(freqs))))
            if sid in self.kept['wavelet'].keys():
                self.wavelets = self.kept['wavelet'][sid]
            else:
                self.make_wavelets(freqs)
                if ((self.cache_limit is None)
                        or self.cache_limit > len(self.kept['wavelet'])):
                    self.kept['wavelet'].update({sid: self.wavelets})
        logger.info('Applying convolve.')
        return ncp.array([ncp.convolve(w, wave, 'same')
                          for w in cast(Array, self.wavelets)])

class WaveletMultiplier(WaveletGenerator):
    """
    CWT class which performs cwt by fft.
    """

    def cwt_fft(self, wave: Array, freqs: Numbers,
                reuse_wavelets: bool = True, logger: Logger = logger) -> Array:
        '''cwt
        Run CWT based on fft.

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
        remake_plan = False
        if self.cuda:
            wave = wave.astype(cp.complex)
        # This if statement can not be method, for performance.
        if (not reuse_wavelets) or (self.fft_wavelets is None):
            if self.cuda:
                remake_plan = True
            sid = ''.join((str(wave.shape), str(id(freqs))))
            if sid in self.kept['fft'].keys():
                self.fft_wavelets = self.kept['fft'][sid]
            else:
                self.make_fft_wavelets(freqs, wave.shape[0] / self.sfreq)
                if((self.cache_limit is None) or
                   self.cache_limit > len(self.kept['fft'])):
                    self.kept['fft'].update({sid: self.fft_wavelets})
        ncp = cp if self.cuda else np
        self._freqs = freqs
        logger.info('Applying FFT mul.')
        # This 4 lines makes this fast a little.
        if self.cuda:
            if remake_plan:
                self.fft_plan = cx_fft.get_fft_plan(wave, axes=(0,))
            to_ifft = self.fft_wavelets * cx_fft.fft(wave, plan=self.fft_plan)
            if remake_plan:
                self.ifft_plan = cx_fft.get_fft_plan(to_ifft, axes=(1,))
            return cx_fft.ifft(to_ifft, plan=self.ifft_plan)
        return ncp.fft.ifft(self.fft_wavelets * ncp.fft.fft(wave))

    def fourier_cwt(self, wave: Array, freqs: Numbers,
                reuse_wavelets: bool = True, logger: Logger = logger) -> Array:
        """
        CWT for wave which is already fourier transformed.
        """
        remake_plan = False
        self.wave_length = wave.shape[0]
        if self.cuda:
            wave = wave.astype(cp.complex)
        # This if statement can not be method, for performance.
        if (not reuse_wavelets) or (self.fft_wavelets is None):
            if self.cuda:
                remake_plan = True
            sid = ''.join((str(wave.shape), str(id(freqs))))
            if sid in self.kept['fft'].keys():
                self.fft_wavelets = self.kept['fft'][sid]
            else:
                self.make_fft_wavelets(freqs, wave.shape[0] / self.sfreq)
                if ((self.cache_limit is None) or
                    self.cache_limit > len(self.kept['fft'])):
                    self.kept['fft'].update({sid: self.fft_wavelets})
        ncp = cp if self.cuda else np
        self._freqs = freqs
        logger.info('Applying FFT mul.')
        # This 4 lines makes this fast a little.
        # if self.cuda:
        #     if remake_plan:
        #         self.ifft_plan = cx_fft.get_fft_plan(wave, axes=(1,))
        #     return cx_fft.ifft(wave, plan=self.ifft_plan)
        return ncp.fft.ifft(self.fft_wavelets * wave)

class WaveletBase(WaveletConvolver, WaveletMultiplier):
    '''
    Base class of wavelets.
    You need to write methods to make single wavelet.
    '''

    def cwt(self, wave: Array, freqs: Union[Numbers, None],
            logger: Logger = logger) -> Array:
        '''Perform CWT

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.
        logger: Logger
            logger

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        reuse_wavelets = False
        if (self.wave_length == wave.shape[-1]) and (self.freqs is freqs):
            reuse_wavelets = True
        self.freqs = freqs
        self.wave_length = wave.shape[-1]
        if self.mode in [CWTMode.Fast, CWTMode.Normal]:
            if isinstance(wave, cp.ndarray):
                logger.info('Cuda is enabled.')
                self.cuda = True
            return self.cwt_fft(wave, freqs, reuse_wavelets, logger)
        if self.cuda:
            logger.warn('''
Cuda is disabled, because cupy cannot convolve in this version.
Numpy will be used.''')
            self.cuda = False
        if isinstance(wave, cp.ndarray):
            logger.error('''
Cuda is disabled, but the wave is cupy.ndarray.
In this version, in Convolve mode, cuda is disabled.
Converting to numpy is too slow. Exit.''')
            raise TypeError('Normal mode cannot use cupy.')
        return self.cwt_convolve(wave, freqs, reuse_wavelets, logger)

    def power(self, wave: Array, freqs: Array,
              logger: Logger = logger) -> Array:
        '''Perform CWT and calcurate power.

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.
        logger: Logger
            logger

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        logger.info('Calculating power.')
        return self.abs(wave, freqs, logger) ** 2

    def abs(self, wave: Array, freqs: Array,
            logger: Logger = logger) -> Array:
        '''Perform CWT and calcurate absolute value.

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.
        logger: Logger
            logger

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        logger.info('Calculating absolute.')
        ncp = cp if self.cuda else np
        return ncp.abs(self.cwt(wave, freqs, logger))

    def clear_cache(self) -> 'WaveletBase':
        '''
        Just clears cache, and returns self.
        '''
        self.kept = {'fft': {}, 'wavelet': {}}
        return self

    def plot(self, freq: float, show: bool = True,
             logger: Logger = logger) -> plt.figure:
        '''
        Plot a wavelet.
        It is just for visualize.

        Parameters
        ----------
        freq: float
            Frequency.
        show: bool
            Show wavelet or not
        logger: Logger
            logger

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        logger.info('Plotting wavelet')
        return plot_wavelet(self, freq, show)

    def set_mode(self, mode: CWTMode,
                 logger: Logger = logger) -> 'WaveletBase':
        '''
        Just set mode.
        >>> morse = Morse().set_mode(CWTMode.Fast)

        Parameters
        ----------
        mode: CWTMode
            Mode of wavelets. It should be imported from ninwavelets.
        logger: Logger
            logger

        Returns
        ---------
        Wavelet instance itself
        '''
        self.mode = mode
        logger.info(f'CWT mode was set to {mode.name}')
        return self

def plot_wavelet(wavelet_obj: WaveletBase, freq: float,
                 show: bool = True, logger: Logger = logger) -> plt.figure:
    '''
    Plot wavelet.

    Parameters
    ----------
    freq: float
        Frequency of Wavelet.
    show: bool
        Show plot.

    Returns
    -------
    Fig of matplotlib.
    '''
    logger.info('Plotting wavelet.')
    freqs = np.array([freq])
    plt_num = 3 if wavelet_obj.help else 2
    wavelet = wavelet_obj.make_wavelet(freqs)
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(plt_num, 1, 1)
    ax.plot(np.arange(0, wavelet.shape[0], 1), wavelet, label='morse')
    ax1 = fig.add_subplot(plt_num, 1, 2, projection='3d')
    ax1.scatter3D(wavelet.real, np.arange(0, wavelet.shape[0], 1),
                  wavelet.imag, label='morse')
    ax.set_title('Generalized Morse Wavelet')
    if plt_num == 3:
        ax2 = fig.add_subplot(313)
        ax2.set_title('Caution')
        ax2.text(0.05, 0.1, wavelet_obj.help)
        ax2.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
    if show:
        plt.show()
    return fig


def plot_tf(data: Array, sfreq: float = 1000,
            vmin: Optional[float] = None, vmax: Optional[float] = None,
            ylabels: Optional[Array] = None,
            cmap: str = 'RdBu_r', show: bool = True,
            logger: Logger = logger) -> plt.Axes:
    '''
    Plot by matplotlib.
    vrange: Tuple[float, float]
        This is range of color.
        Same as tuple of vmin and vmax of matplotlib.
    '''
    logger.info('Plotting time-frequency map')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if ylabels:
        ax.set_yticks(np.arange(0, data.shape[0],
                                data.shape[0] / ylabels.shape))
        ax.set_yticklabels(ylabels)
    image = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(image, cax=ax_cb)
    if show:
        plt.show()
    return ax
