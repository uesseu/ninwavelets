import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifft, fft
from scipy import fftpack
from typing import Union, List, Iterator, Callable, Tuple, cast, Optional, Dict
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial, reduce
from operator import mul
from logging import getLogger, INFO, basicConfig, NullHandler, Logger

logger = getLogger('ninwavelets')
logger.addHandler(NullHandler())

try:
    import cupy as cp
except ImportError as error:
    print(error)
    print('Cupy could not be loaded.')

Numbers = Union[List[float], np.ndarray, range]
Array = Union[np.ndarray, cp.ndarray]
Float = Optional[float]
Floats = Optional[Tuple[float, float]]
NORM_CONSTANT = np.sqrt(0.5)



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


class WaveletBase:
    '''
    Base class of wavelets.
    You need to write methods to make single wavelet.
    '''

    def __init__(self, sfreq: float = 1000, real_wave_length: float = 1.,
                 cuda: bool = False, keep_num: Optional[int] = None) -> None:
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
        '''
        self.mode: CWTMode = CWTMode.Fast
        self.sfreq: float = sfreq
        self.help: str = ''
        self.real_wave_length: float = real_wave_length
        self.freq_dist: float
        self.cuda: bool = cuda
        self._freqs: Numbers = None
        self.freqs: Optional[Array] = None
        self.kept: Dict[str, Dict[str, Array]] = {'fft': {}, 'wavelet': {}}
        self.keep_num = keep_num

    def _setup_trans_shape(self, freq: float,
                           real_wave_length: float) -> Array:
        '''
        Setup wave shape.
        real_length is length of wavelet(for example, sec or msec)
        self.real_wave_length is length of wave to analyze.

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
        result = ncp.arange(0, self.sfreq * real_wave_length, 1) / freq
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
        total: float = self.real_wave_length
        one: float = 1 / self.sfreq
        if zero_mean:
            result = np.arange(-total / 2, total / 2, one) * 2 * freq * np.pi / self.peak_freq(freq)
        else:
            result = np.arange(0., total, one) * 2 * freq * np.pi / self.peak_freq(freq)
        return cp.asarray(result, np.float64) if self.cuda else result

    def peak_freq(self, freq: float) -> float:
        return 1.

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
            t = self._setup_trans_shape(real_length, real_length)
            result = formula(t, freq)
            return result / self.get_wavelet_norm(ncp.fft.ifft(result), (1,))
        else:
            wavelet = self.make_wavelet(freq)
            half = int((self.sfreq * self.real_wave_length
                        - wavelet.shape[0]) / 2)
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
            t = ncp.tile(self._setup_trans_shape(real_length, real_length), (freqs.shape[0], 1))
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
            # It is slower code. In this case, using tuple was fast.

            # wavelet = self.make_wavelets(freqs)
            # half = int((self.sfreq * self.real_wave_length
            #             - wavelet.shape[0]) / 2)
            # wavelet = ncp.hstack((ncp.zeros((wavelet.shape[0], half)),
            #                       wavelet, ncp.zeros((wavelet.shape[0], half))))
            # result = ncp.fft.fft(wavelet)
            # result.imag = ncp.abs(result.imag)
            # result.real = ncp.abs(result.real)
            # return result

            self.freq_dist = freqs[1] - freqs[0]
            make_w = partial(self.make_fft_wavelet, real_length=real_length)
            self.fft_wavelets = ncp.array(tuple(map(make_w, freqs)))
            return self.fft_wavelets

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

    def get_wavelet_norm(self, wavelet: Array, axis: Optional[tuple] = None) -> Array:
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
            timeline = self._setup_trans_shape(freq, self.real_wave_length)
            if self.cuda:
                wavelet = cp.fft.ifft(self.cp_trans_formula(timeline))
            else:
                wavelet = ifft(self.trans_formula(timeline))
            half = int(wavelet.shape[0])
            start, stop = half // 2, half // 2 * 3
            total_wavelet = ncp.hstack((ncp.conj(ncp.flip(wavelet)), wavelet))
            wavelet = total_wavelet[start: stop]
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
            timelines = ncp.array(tuple(self._setup_trans_shape(freq, self.real_wave_length)
                              for freq in freqs))
            if self.cuda:
                wavelet = cp.fft.ifft(self.cp_trans_formula(timelines))
            else:
                wavelet = ifft(self.trans_formula(timelines))
            half = int(wavelet.shape[0])
            start, stop = half // 2, half // 2 * 3
            total_wavelet = ncp.hstack((ncp.conj(ncp.flip(wavelet)), wavelet))
            wavelet = total_wavelet[start: stop]
            # wavelet /= self.get_wavelet_norm(wavelet)
            divs = ncp.array(self.get_wavelet_norm(wavelet, (1,)))
            wavelet /= ncp.tile(divs, (wavelet.shape[1], 1)).T
        else:
            timeline = ncp.array([self._setup_waveletshape(freq, 1, zero_mean=True) for freq in freqs])
            formula = self.cp_formula if self.cuda else self.formula
            wavelet = formula(timeline, freqs)
            divs = ncp.array(self.get_wavelet_norm(wavelet, (1,)))
            wavelet /= ncp.tile(divs, (wavelet.shape[1], 1)).T
        self.wavelets = wavelet
        return wavelet


    def cwt(self, wave: Array, freqs: Union[Numbers, None],
            logger: Logger = logger) -> Array:
        '''Perform CWT

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        reuse_wavelets = False
        if (self.real_wave_length == wave.shape[0] / self.sfreq) and\
                (self.freqs is freqs):
            reuse_wavelets = True
        self.freqs = freqs
        self.real_wave_length = wave.shape[0] / self.sfreq
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

    def cwt_convolve(self, wave: Array, freqs: Numbers,
                     reuse_wavelets: bool, logger: Logger = logger) -> Array:
        '''
        Backend of cwt in convolve mode

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.

        Returns
        ----------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        ncp = cp if self.cuda else np
        if (not reuse_wavelets) or (not hasattr(self, 'current_wavelets')):
            sid = ''.join((str(wave.shape), str(id(freqs))))
            if sid in self.kept['wavelet'].keys():
                self.current_wavelets = self.kept['wavelet'][sid]
            else:
                self.make_wavelets(freqs)
                self.current_wavelets = self.wavelets
                if (self.keep_num is None) or self.keep_num > len(self.kept['wavelet']):
                    self.kept['wavelet'].update({sid: self.current_wavelets})
        logger.info('Applying convolve.')
        return ncp.array([ncp.convolve(w, wave, 'same')
                          for w in self.current_wavelets])

    def cwt_fft(self, wave: Array, freqs: Numbers,
                reuse_wavelets: bool = True, logger: Logger = logger) -> Array:
        '''cwt
        Run CWT.

        wave: np.ndarray
            Wave to analyze
        freqs: Union[List[float], range, np.ndarray]
            Frequencies. It can be argument of cwt, but it is slow.
            If you want to calculate repeatedly, you should run
            make_fft_wavelets before cwt, and freqs should be None.
        max_freq: int
            Max Frequency
        reuse: bool
            Use wavelet which was made before.

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        if (not reuse_wavelets) or (not hasattr(self, 'current_fft_wavelets')):
            sid = ''.join((str(wave.shape), str(id(freqs))))
            if sid in self.kept['fft'].keys():
                self.current_fft_wavelets = self.kept['fft'][sid]
            else:
                self.make_fft_wavelets(freqs, wave.shape[0] / self.sfreq)
                self.current_fft_wavelets = self.fft_wavelets
                if(self.keep_num is None) or self.keep_num > len(self.kept['fft']):
                    self.kept['fft'].update({sid: self.current_fft_wavelets})
        fft = cp.fft.ifft if self.cuda else fftpack.ifft
        ifft = cp.fft.fft if self.cuda else fftpack.fft
        self._freqs = freqs
        logger.info('Applying FFT mul.')
        return ifft(self.current_fft_wavelets * fft(wave))

    def power(self, wave: Array, freqs: Array,
              logger: Logger = logger) -> Array:
        '''Perform CWT and calcurate power.

        Parameters
        ----------
        wave: Union[np.ndarray, cp.ndarray]
            Raw wave to transform.
        freqs: List[float]
            Frequencies.

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

        Returns
        -------
        Result of CWT: Union[np.ndarray, cp.ndarray]
        '''
        logger.info('Calculating absolute.')
        ncp = cp if self.cuda else np
        return ncp.abs(self.cwt(wave, freqs, logger))

    def plot(self, freq: float, show: bool = True,
             logger: Logger = logger) -> plt.figure:
        logger.info('Plotting wavelet')
        return plot_wavelet(self, freq, show)

    def set_mode(self, mode: CWTMode, logger: Logger = logger) -> 'WaveletBase':
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
    wavelet = wavelet_obj.make_wavelets(freqs)[0]
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
    ax.set_aspect('auto')
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
