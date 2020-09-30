import numpy as np
import cupy as cp
from mpl_toolkits.mplot3d import Axes3D
from typing import Any
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from mne.time_frequency.tfr import morlet, cwt
from ninwavelets import (Morse, Morlet, CWTMode, np2cp,
                         Haar, plot_tf, MexicanHat, Shannon, Baseline)
from mne.io import Raw
from mne import verbose
import gc
from time import time
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
    morse = morse_obj.make_wavelet(hz)
    nm = Morlet(sfreq)
    nm.mode = CWTMode.Normal
    nin_morlet = nm.make_wavelet(hz)

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
    ax.plot(Haar(1000).make_wavelet(hz), label='Haar Wavelet')

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


def cwt_test(cuda: bool = False, show: bool = False, random: bool = True) -> None:
    min_freq = 30
    max_freq = 500
    sin = make_example(4, cuda, random)
    if cuda:
        sin = cp.asarray(sin)

    ncp = cp if cuda else np
    log.info('''Fast mode test for GMW''')
    t = time()
    morse = Morse(cuda=cuda, sfreq=1000)
    result_morse = morse.power(sin, ncp.arange(min_freq, max_freq, 1))
    print(f'Morse {time() - t}')

    log.info('''Change to Normal mode test for GMW only for numpy''')
    if not cuda:
        morse.mode = CWTMode.Normal
    morse.power(sin, ncp.arange(min_freq, max_freq, 1))

    log.info('''Normal mode test
Normal mode is only for numpy
Because cupy 7.6.0 has no method named convolve''')
    nin_morlet = Morlet(cuda=False, sfreq=1000)
    nin_morlet.mode = CWTMode.Convolve
    t = time()
    result_morlet = nin_morlet.power(cp.asnumpy(sin), np.arange(min_freq, max_freq, 1))
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
        ax1.imshow(np.abs(result_morse), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax2.imshow(np.abs(result_morlet), cmap='RdBu_r', vmin=vmin, vmax=vmax)
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

    plot_tf(result_morse)
    # plt.show()


def other_wavelet_test() -> None:
    hz = 10
    s = 7
    mexcan = MexicanHat().make_wavelet(hz)
    shannon = Shannon().make_wavelet(hz)
    morlet = Morlet(sigma=s).make_wavelet(hz)
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
    fig = plt.figure()
    w = morse.make_wavelet(hz)
    a = morse.make_fft_wavelet(hz)
    b = nin_morlet.make_wavelet(hz)
    c = nin_morlet.make_fft_wavelet(hz)
    d = morlet(1000, [hz])[0]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(w, label='Generalized Morse wavelet')
    ax.plot(a, label='FFTed Generalized Morse wavelet')
    ax.plot(b, label='Morlet wavelet')
    ax.plot(b.imag, label='Morlet wavelet')
    ax.plot(np.abs(c.real), label='FFTed Morlet wavelet')
    ax.plot(np.abs(fft(d)), label='MNE morlet')
    ax.plot(c.imag, label='imag of FFTed Morlet wavelet')
    ax.plot(normal_morlet.make_wavelet(hz), label='Morlet Wavelet Normal Mode')
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


def speed_test(i: int) -> None:
    length = 1
    repeat = 10
    reg = 10
    t = time()
    sin = make_example(length, False)
    c_sin = make_example(length, True)
    wv = make_example(length, False)
    wv_mne = np.array([make_example(length, False)])
    freqs = np.arange(30, 500, 1)
    c_freqs = cp.arange(30, 500, 1)


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
        result_mne = np.abs(cwt(wv_mne, mne_morlet)[0]) ** 2
    mne_time = time() - t
    print(f'MNE morlet {mne_time}')

    #====================
    # PyWavelet
    #====================
    import pywt
    t = time()
    for n in range(int(repeat/10)):
        cwtmatr, result_freqs = pywt.cwt(wv, freqs, 'cmor1.5-1.0')
        np.abs(cwtmatr) ** 2
    pywavelet_time = (time() - t) * 10
    print(f'PyWavelet morlet {pywavelet_time}')

    #====================
    # NinWavelet
    #====================
    t = time()
    nin_morlet = Morlet(cuda=True, sfreq=1000)
    for n in range(repeat):
        result_morlet = nin_morlet.power(c_sin, freqs)
    ninwavelet_time_cuda = time() - t
    print(f'Ninwavelets cuda morlet {ninwavelet_time_cuda}')

    t = time()
    nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=0)
    for n in range(repeat):
        result_morlet = nin_morlet.power(sin, np.arange(30, 500, 1))
    ninwavelet_time_slow = time() - t
    print(f'Ninwavelets slow morlet {ninwavelet_time_slow}')

    t = time()
    nin_morlet = Morlet(cuda=False, sfreq=1000)
    for n in range(repeat):
        result_morlet_cuda = nin_morlet.power(sin, freqs)
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

    plt.bar(np.arange(0, 7, 1),
            1 / np.array([pywavelet_time, scipy_time, mne_time, swan_time, ninwavelet_time_slow, ninwavelet_time, ninwavelet_time_cuda]),
            tick_label=['PyWavelet', 'Scipy', 'MNE', 'Swan', 'Ninwavelets\nSlow', 'Ninwavelets', 'Ninwavelets\nCuda'])
    plt.xlabel('Packages')
    plt.ylabel(f'Speed. ({repeat}trial / sec) Bigger is fast.')
    plt.title(f'1sec wave, Sampling frequency:1000Hz\nMorletWavelet(30~500Hz) {repeat}times')
    plt.show()

def geom_test():
    morse = Morse()
    result = morse.power(np.random.random(1000), np.geomspace(1, 10, 1000))
    plt.imshow(result)
    plt.show()

def tune(wave):
    morse = Morse()
    for n in range(10):
        result = morse.power(wave, freqs)

def tune_cuda(wave):
    morse = Morse(cuda=True)
    for n in range(10):
        result = morse.power(wave, freqs)

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
        other_wavelet_test()
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
        data = [np.asarray(np.arange(1, 1000, 1)) for i in range(100)]
        t = time()
        for n in data:
            cp.asarray(n)
        print(f'Sync {time() - t}')

        t = time()
        for n in range(100):
            np2cp(*data)
        print(f'Async {time() - t}')
    if 'geom' in argv:
        geom_test()
    if 'tune' in argv:
        wave = np.random.random(1000)
        cp_wave = cp.random.random(1000)
        freqs = np.geomspace(1, 10, 1000)
        tune(wave)
        tune_cuda(cp_wave)
