import numpy as np
import cupy as cp
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Union
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from mne.time_frequency.tfr import morlet, cwt
from ninwavelets import (Morse, Morlet, CWTMode, np2cp,
                         Haar, plot_tf, MexicanHat, Shannon, Baseline,
                         windows, experimental)
from mne.io import Raw
from mne import verbose
import gc
from time import time, sleep
from sys import argv
from logging import getLogger, INFO, WARNING, basicConfig, ERROR, CRITICAL
from functools import partial
from scipy import signal
from contextlib import redirect_stdout
import os
import seaborn as sns
sns.set(font_scale=2)

basicConfig(level=WARNING)
log = getLogger()
Array = Union[np.ndarray, cp.ndarray]


def make_example(length: float = 3, cuda: bool = True, random: bool = True) -> np.ndarray:
    freq: float = 60
    ncp = cp if cuda else np
    time: ncp.ndarray = ncp.arange(0, length, 0.001)
    if random:
        return ncp.random.random(length * 1000)
    sin = ncp.array(ncp.sin(time * freq * 2 * ncp.pi) +
                   ncp.sin(time * 160 * 2 * ncp.pi) * ncp.sin(time * ncp.pi) +
                   ncp.sin(ncp.pad(ncp.arange(0, length / 2, 0.001),
                                 [int(length * 250), int(length * 250)],
                                 'constant') *
                          300 * 2 * ncp.pi)
                   )
    # result = [ncp.copy(sin) for n in range(1000)]
    return sin


def test() -> None:
    morse = Morse(1000, 17.5, 3)
    freq = 60
    time = np.arange(0, 3, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi))
    result = morse.power(sin, np.arange(1, 100, 1))
    plt.imshow(result, cmap='RdBu_r')
    plt.gca().invert_yaxis()
    plt.title('CWT of 60Hz sin wave')
    plt.show()


def test3d() -> None:
    sfreq = 1000
    hz = 20
    go = morlet(sfreq, [hz])[0]
    mm = morlet(sfreq, [hz], zero_mean=True)[0]
    morse_obj = Morse(sfreq, 17.5, 3)
    morse = morse_obj.make_wavelets([hz])[0]
    nm = Morlet(sfreq)
    nm.mode = CWTMode.Normal
    nin_morlet = nm.make_wavelets([hz])[0]

    half_morse = morse.shape[0] / 2
    morse_time = np.arange(-half_morse, half_morse, 1)
    half_mm = mm.shape[0] / 2
    morlet_time = np.arange(-half_mm, half_mm, 1)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    print(np.linalg.norm(morse))
    print(np.linalg.norm(mm))
    print(np.linalg.norm(nin_morlet))

    ax.plot(morse_time, morse, label='Morse Wavelet')
    ax.plot(morse_time, nin_morlet, label='Morlet Wavelet')
    ax.plot(morse_time, morse.imag, label='Morse Imag')
    ax.plot(morlet_time, mm, label='MNE Morlet')
    ax.plot(morlet_time, mm.imag, label='MNE Morlet imag')
    ax.plot(morlet_time, go, label='Gabor Wavelet')
    ax.plot(Haar(1000).make_wavelets([hz])[0], label='Haar Wavelet')

    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter3D(morse.real, morse_time, morse.imag, label='morse')
    ax1.scatter3D(mm.real, morlet_time, mm.imag, label='MNE morlet')
    ax1.scatter3D(go.real, morlet_time, go.imag, label='gobar')
    handler, label = ax.get_legend_handles_labels()
    handler1, label1 = ax1.get_legend_handles_labels()
    ax.legend(label+label1, loc='upper right')
    ax.set_title('morse and morlet')
    plt.show()


def plot_sin_fft() -> None:
    freq = 60
    time = np.arange(0, 0.3, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi))
    time2 = np.arange(0, 0.6, 0.001)
    sin2 = np.array(np.sin(time2 * freq * 2 * np.pi))
    plt.plot(sin)
    plt.plot(sin2)
    plt.show()
    plt.plot(np.abs(fft(sin)))
    plt.plot(np.abs(fft(sin2)))
    plt.show()


def cwt_test(cuda: bool = False, show: bool = False, random: bool = False) -> None:
    min_freq = 30
    max_freq = 500
    sin = make_example(1, cuda, random)
    if cuda:
        sin = cp.asarray(sin)

    ncp = cp if cuda else np
    log.info('''Fast mode test for GMW''')
    t = time()
    morse = Morse(cuda=cuda, sfreq=1000)
    result_morse = ncp.square(ncp.abs(
        morse.cwt(cp.array([sin]), ncp.arange(min_freq, max_freq, 1))))
    print(f'Morse {time() - t}')

    log.info('''Change to Normal mode test for GMW only for numpy''')
    if not cuda:
        morse.mode = CWTMode.Normal

    log.info('''Normal mode test
Normal mode is only for numpy
Because cupy 7.6.0 has no method named convolve''')
    nin_morlet = Morlet(cuda=False, sfreq=1000)
    nin_morlet.mode = CWTMode.Convolve
    t = time()
    result_morlet = np.square(np.abs(
        nin_morlet.cwt(cp.asnumpy([sin]), np.arange(min_freq, max_freq, 1))))
    print(f'Morlet {time() - t}')
    if cuda:
        sin = cp.asnumpy(sin)
        nin_morlet = morlet(1000, np.arange(min_freq, max_freq, 1), zero_mean=True)[0]
        # morlet = Morlet(cuda=False)
        # morlet.mode = CWTMode.Fast
    else:
        sin = cp.asnumpy(sin)
        # morlet = Morlet(cuda=False)
    t = time()
    result_mne = cwt(np.array([sin]),
                     morlet(1000, np.arange(min_freq, max_freq, 1)),
                     use_fft=True)[0] ** 2
    print(f'MNE {time() - t}')

    if show:
        # plt.plot(nin_morlet.wavelets[15])
        plt.show()
        if cuda:
            result_morse = cp.asnumpy(result_morse)
            result_morlet = cp.asnumpy(result_morlet)
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        vmin = 0
        vmax = 10
        ax1.imshow(cp.asnumpy(cp.abs(result_morse[0])), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax2.imshow(np.abs(result_morlet[0]), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax3.imshow(np.abs(result_mne), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax1.set_title('Morse')
        ax2.set_title('Morlet')
        ax3.set_title('MNE')
        plt.show()
    print(f'Morse max if {np.max(np.abs(result_morse))}')
    print(f'Morlet max is {np.max(np.abs(result_morlet))}')
    print(f'MNE max is {np.max(np.abs(result_mne))}')
    print(f'Morse mean is {np.abs(result_morse).mean()}')
    print(f'Morlet mean is {np.abs(result_morlet).mean()}')
    print(f'MNE mean is {np.abs(result_mne).mean()}')
    # result_morse = morse.power(sin, reuse=True)

    plot_tf(cp.asnumpy(result_morse[0]))
    # plt.show()


def other_wavelet_test() -> None:
    hz = 10
    s = 7
    mexcan = MexicanHat().make_wavelets([hz])
    shannon = Shannon().make_wavelets([hz])
    morlet = Morlet(sigma=s).make_wavelets([hz])
    plt.plot(mexcan)
    plt.plot(shannon)
    plt.plot(morlet)
    plt.show()
    plt.plot(np.abs(fft(shannon)))
    plt.plot(np.abs(fft(morlet)))
    plt.show()
    sin = make_example(1, False)

    log.info('Other wavelets')
    log.info('Haar wavelets')
    result_haar = Haar(1000).power(sin, np.arange(1, 1000, 1))
    plot_tf(result_haar)

    log.info('Other wavelets')
    log.info('Mexican wavelets')
    result_mexican = MexicanHat(1000).power(sin, np.arange(1, 1000, 1))
    plot_tf(result_mexican)


def fft_wavelet_test() -> None:
    hz = 10.
    r = 3
    b = 17.5
    s = 7
    morse = Morse(r=r, b=b)
    nin_morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet.mode = CWTMode.Normal
    w = morse.make_wavelets([hz])
    a = morse.make_fft_wavelets([hz])[0]
    b = nin_morlet.make_wavelets([hz])[0]
    c = nin_morlet.make_fft_wavelets([hz])[0]
    d = morlet(1000, [hz])[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(w, label='Generalized Morse wavelet')
    ax.plot(a, label='FFTed Generalized Morse wavelet')
    ax.plot(b, label='Morlet wavelet')
    ax.plot(b.imag, label='Morlet wavelet')
    ax.plot(np.abs(c.real), label='FFTed Morlet wavelet')
    ax.plot(np.abs(fft(d)), label='MNE morlet')
    ax.plot(c.imag, label='imag of FFTed Morlet wavelet')
    ax.plot(normal_morlet.make_wavelets([hz]), label='Morlet Wavelet Normal Mode')
    handler, label = ax.get_legend_handles_labels()
    ax.legend(label, loc='upper right')
    plt.show()


def eeg(cuda: bool) -> None:
    '''
    This test code reads my eeg.
    I am not sure whether I can open my eeg.
    My boss may says "You shoulnt!"
    If you have your own eeg, why dont you process your eeg?
    '''
    raw = Raw('/home/ninja/ninja.fif')
    data = raw.get_data()[raw.ch_names.index('EEG O1-Ref')]
    d = data[150*500: 190*500]
    d = Baseline(d, 1000, 0, 1).zscore()
    tf = Morse(raw.info['sfreq'], cuda=True).power(d, np.arange(0.1, 50, 0.1))
    ax = plot_tf(tf, frange=(0, 50, 10), trange=(0, 40, 10), sfreq=500,
                 show=False)
    ax.set_title('My EEG power(O1)')
    ax.set_xlabel('Time Course(sec)')
    ax.set_ylabel('Hz')
    plt.show()

def speed_wavelet_test() -> None:
    freq_range = 30, 500
    freqs = np.arange(freq_range[0], freq_range[1], 1)
    t = time()
    for n in range(10):
        mne_morlet = morlet(1000, freqs)
    print('MNE wavelet generation: ', time() - t)

    t = time()
    for n in range(10):
        nin_morlet = Morlet(1000,gabor=True).make_fft_wavelets(freqs)
    print('Nin wavelet generation: ', time() - t)

def speed_test(i: int) -> None:
    length = 1
    repeat = 100
    freq_range = 30, 500
    t = time()
    sin = make_example(length, False, False)
    c_sin = make_example(length, True, False)
    wv = make_example(length, False)
    wv_mne = np.array([make_example(length, False)])
    freqs = np.arange(freq_range[0], freq_range[1], 1)


    #====================
    # Scipy
    #====================
    from scipy import signal
    t = time()
    for n in range(repeat):
        result_sci = np.abs(signal.cwt(wv, signal.morlet2, freqs)) ** 2
    scipy_time = time() - t
    print(f'Scipy morlet {scipy_time}')

    #====================
    # MNE
    #====================
    t = time()
    mne_morlet = morlet(1000, freqs)
    for n in range(repeat):
        result_mne = np.square(np.abs(cwt(wv_mne, mne_morlet)[0]))
    mne_time = time() - t
    print(f'MNE morlet {mne_time}')

    #====================
    # NinWavelet
    #====================
    sp_c_sin = np.array([sin for n in range(repeat)])
    nin_morlet = Morlet(cuda=True, sfreq=1000)
    t = time()
    result_morlet_cuda = nin_morlet.power(cp.asarray(sp_c_sin), freqs)
    nin_morlet_tmp = Morlet(cuda=False, sfreq=1000)
    for n in range(int(repeat / 10)):
        result_morlet = nin_morlet_tmp.power(sin, freqs)
    print(result_morlet_cuda[-1, -1, -1])
    ninwavelet_time_cuda = time() - t
    print(f'Ninwavelets cuda morlet {ninwavelet_time_cuda}')

    nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=0)
    t = time()
    for n in range(repeat):
        result_morlet = nin_morlet.power(sin, np.arange(freq_range[0], freq_range[1], 1))
    ninwavelet_time_slow = time() - t
    print(f'Ninwavelets slow morlet {ninwavelet_time_slow}')

    nin_morlet = Morlet(cuda=False, sfreq=1000)
    sp_sin = np.array([sin for n in range(repeat)])
    t = time()
    for n in range(repeat):
        result_morlet = nin_morlet.power(sin, freqs)
    # result_morlet = nin_morlet.power(sp_sin, freqs)
    ninwavelet_time = time() - t
    print(f'Ninwavelets numpy morlet {ninwavelet_time}')
    #====================
    # SWAN
    #====================
    from swan import pycwt
    swan_morlet = pycwt.Morlet()
    t = time()
    for n in range(repeat):
        r = np.abs(pycwt.cwt_f(wv, freqs, 1000, swan_morlet)) ** 2
    swan_time = time() - t
    print(f'Swan morlet {swan_time}')

    plt.bar(np.arange(0, 6, 1),
            1 / np.array([scipy_time, mne_time, swan_time, ninwavelet_time_slow, ninwavelet_time, ninwavelet_time_cuda]),
            tick_label=['Scipy', 'MNE', 'Swan', 'Ninwavelets\nNaive', 'Ninwavelets\nCached', 'Ninwavelets\nCuda'])
    plt.xlabel('Packages')
    plt.ylabel(f'Speed. ({repeat}trial / sec) Bigger is fast.')
    plt.title(f'1sec wave, Sampling frequency:1000Hz\nMorletWavelet({freq_range[0]}~{freq_range[1]}Hz) {repeat}times')
    plt.show()

def geom_test() -> None:
    morse = Morse()
    result = morse.power(np.random.random(1000), np.geomspace(1, 10, 1000))
    plt.imshow(result)
    plt.show()

def tune(wave: Array) -> None:
    morse = Morse()
    for n in range(10):
        result = morse.power(wave, freqs)

def tune_cuda(wave: Array, repeat: int) -> None:
    morse = Morse(cuda=True)
    for n in range(repeat):
        result = morse.power(wave, freqs)

def tune_2d(wave: Array) -> None:
    morse = Morse(cuda=True)
    result = morse.power(wave, freqs)

def test_2d() -> None:
    morse = Morse()
    wv = cp.array([make_example(1, True) for n in range(2)])
    freqs = np.arange(30, 50, 1)
    morse.cwt(wv, freqs)
    morse.cwt(wv[0], freqs)


if __name__ == '__main__':
    print('Test Run')
    # plot_sin_fft()
    # test()
    cuda = True if 'cuda' in argv else False
    if cuda:
        print('CUDA is on')
    if 'sin' in argv:
        plot_sin_fft()
    if 'wave' in argv:
        test3d()
        fft_wavelet_test()
        # other_wavelet_test()
    if '2d' in argv:
        test_2d()
    if 'cwt' in argv:
        cwt_test(cuda, show=True)
    if 'eeg' in argv:
        eeg(cuda)
    if 'speed' in argv:
        if 'multi' in argv:
            with Pool(6) as p:
                p.map(partial(speed_test, cuda=cuda), range(8))
        else:
            speed_test(0)
    if 'all' in argv:
        cwt_test(True, show=True)
        cwt_test(False, show=True)
        other_wavelet_test()
    if 'async' in argv:
        data = [np.asarray(np.arange(1, 1000, 1)) for i in range(10)]
        t = time()
        res = tuple(cp.asarray(n) for n in data)
        print(res[0][0])
        print(f'Sync {time() - t}')
        t = time()
        res = np2cp(data, sep=1)
        print(res[0][0])
        print(f'Async {time() - t}')
    if 'geom' in argv:
        geom_test()
    if 'tune' in argv:
        print('tune')
        # wave = np.random.random(1000)
        cp_wave = cp.random.random(100)
        sp_cp_wave = cp.array([cp_wave for n in range(500)])
        freqs = np.geomspace(1, 10, 1000)
        # tune(wave)
        tune_cuda(cp_wave, 500)
        tune_2d(sp_cp_wave)
    if 'wavelet_speed' in argv:
        speed_wavelet_test()
    if 'window' in argv:
        wave = np.ones(1000)
        plt.plot(windows.hanning_window(1000, cuda=False, tukey_ratio=0.1) * wave)
        plt.plot(np.abs(ifft(wave)))
        plt.show()
    if 'app' in argv:
        sin = make_example(1, False, False)
        sin10 = np.tile(sin, 100).reshape((100,sin.shape[-1]))
        morse = Morse()
        freqs = np.arange(30, 150)
        res=morse.cwt(sin, freqs)
        t = time()
        for n in range(100):
            res=morse.fourier_cwt(fft(sin), freqs)
        print(time() - t)
        plt.imshow(np.abs(res))
        plt.show()
        morse1 = experimental.Morse()
        t = time()
        res = morse1.app_cwt(sin10, freqs, band_rate=0.1)
        print(time() - t)
        plt.imshow(np.abs(res[50]))
        plt.show()
