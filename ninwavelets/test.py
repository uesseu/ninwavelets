import numpy as np
import cupy as cp
from typing import Union, Optional, List
from multiprocessing import Pool
from threading import Thread

from scipy.fftpack import fft, ifft
from ninwavelets import (Morse, Morlet, CWTMode, np2cp,
                         plot_tf, MexicanHat, Shannon, Baseline,
                         windows)
from time import time
from sys import argv
from logging import getLogger, INFO, WARNING, basicConfig, ERROR, CRITICAL
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = matplotlib.font_manager.FontProperties(fname=font_path, size=18)
matplotlib.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["font.family"] = "IPAexGothic"
# import seaborn as sns
# sns.set(font_scale=2)

basicConfig(level=WARNING)
log = getLogger()
Array = Union[np.ndarray, cp.ndarray]


def make_example(length: float = 3, cuda: bool = True,
                 random: bool = True) -> np.ndarray:
    freq: float = 60
    ncp = cp if cuda else np
    time: Array = ncp.arange(0, length, 0.001)
    if random:
        return ncp.random.random(int(length * 1000))
    sin = ncp.array(ncp.sin(time * freq * 2 * ncp.pi) +
                    ncp.sin(time * 160 * 2 * ncp.pi) * ncp.sin(time * ncp.pi) +
                    ncp.sin(ncp.pad(ncp.arange(0, length / 2, 0.001),
                                    [int(length * 250), int(length * 250)],
                                    'constant') * 300 * 2 * ncp.pi)
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
    from mne.time_frequency import morlet
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


def precision() -> None:
    min_freq = 20
    max_freq = 100
    error = 0
    max_value = []
    min_value = []
    from mne.time_frequency.tfr import morlet, cwt
    for n in range(100):
        sin = make_example(1, False, True)
        mne_morlet = morlet(1000, np.arange(
            min_freq, max_freq, 1), zero_mean=True)[0]
        nin_morlet = Morlet(cuda=False, sfreq=1000)
        result_morlet = np.square(np.abs(
            nin_morlet.cwt(np.array([sin]), np.arange(min_freq, max_freq, 1))))
        result_mne = np.abs(cwt(np.array([sin]),
                                morlet(1000, np.arange(min_freq, max_freq, 1)),
                                use_fft=True)[0] ** 2)
        error += np.count_nonzero(np.abs(result_morlet[0][:, 200: 800] /
                                         result_mne[:, 200: 800] - 1) > 0.001)
        max_value.append(np.max(np.abs(result_morlet[0][:, 200: 800] /
                                       result_mne[:, 200: 800] - 1)))
        min_value.append(np.min(np.abs(result_morlet[0][:, 200: 800] /
                                       result_mne[:, 200: 800] - 1)))
    print(error, 80 * 600 * 100)
    print(error / (80 * 600 * 100))
    print(max(max_value))
    print(min(min_value))


def cwt_test(cuda: bool = False, show: bool = False,
             random: bool = False) -> None:
    cmap = 'rainbow'
    min_freq = 30
    max_freq = 100
    cuda = False
    random = True
    sin = make_example(1, cuda, random)
    if cuda:
        sin = cp.asarray(sin)

    ncp = cp if cuda else np
    log.info('''Fast mode test for GMW''')
    t = time()
    morse = Morse(cuda=cuda, sfreq=1000)
    # morse.mode = CWTMode.Normal
    result_morse = ncp.square(ncp.abs(
        morse.cwt(ncp.array([sin]), ncp.arange(min_freq, max_freq, 1))))
    print(f'Morse {time() - t}')

    log.info('''Change to Normal mode test for GMW only for numpy''')

    log.info('''Normal mode test
Normal mode is only for numpy
Because cupy 7.6.0 has no method named convolve''')
    # nin_morlet = Morlet(cuda=False, sfreq=1000)
    nin_morlet = Morlet(cuda=False, sfreq=1000)
    # nin_morlet.mode = CWTMode.Normal
    t = time()
    result_morlet = np.square(np.abs(
        nin_morlet.cwt(ncp.array([sin]), ncp.arange(min_freq, max_freq, 1))))
    print(f'Morlet {time() - t}')
    from mne.time_frequency.tfr import morlet, cwt
    if cuda:
        sin = cp.asnumpy(sin)
        mne_morlet = morlet(1000, np.arange(
            min_freq, max_freq, 1), zero_mean=True)[0]
        # morlet = Morlet(cuda=False)
        # morlet.mode = CWTMode.Fast
    else:
        sin = cp.asnumpy(sin)
        # morlet = Morlet(cuda=False)
    t = time()
    result_mne = np.abs(cwt(np.array([sin]),
                            morlet(1000, np.arange(min_freq, max_freq, 1)),
                            use_fft=True)[0] ** 2)
    print(f'MNE {time() - t}')
    print('Comparison')
    # plt.imshow((result_morlet[0] - result_mne),
    #            vmin=0, vmax=0.0000005, cmap='rainbow')

    # print(np.sqrt((np.square(result_morlet[0][:, 200: 800]
    #                          - result_mne[:, 200: 800])
    #       / result_morlet[0][:, 200: 800]).max()))

    # plt.imshow(result_morlet[0][:, 200: 800] /
    #            result_mne[: , 200: 800],
    #            cmap='rainbow', vmin = 0.999, vmax=1.001)
    # plt.imshow(np.abs(np.fft.fft(result_morlet[0][:, 200: 800]) /
    #                   np.fft.fft(result_mne[: , 200: 800])),
    #            cmap='rainbow', vmin = 0.9999, vmax=1.0001)
    print(np.abs(np.fft.fft(result_morlet[0])[:, 200: 800] /
                 np.fft.fft(result_mne)[:, 200: 800]).max())
    print(result_morse.shape)
    # plt.show()

    if show:
        # plt.plot(nin_morlet.wavelets[15])
        plt.show()
        import seaborn as sns
        sns.set(font_scale=2)
        if cuda:
            result_morse = cp.asnumpy(result_morse)
            result_morlet = cp.asnumpy(result_morlet)
        fig = plt.figure(figsize=(18, 16))
        axes = [fig.add_subplot(3, 1, n+1) for n in range(3)]
        vmin = 0
        vmax = 1
        ims = []
        ims.append(axes[0].imshow(np.abs(result_morse[0]), cmap=cmap,
                                  vmin=vmin, vmax=vmax))
        ims.append(axes[1].imshow(np.abs(result_morlet[0]), cmap=cmap,
                                  vmin=vmin, vmax=vmax))
        ims.append(axes[2].imshow(np.abs(result_mne), cmap=cmap,
                                  vmin=vmin, vmax=vmax))
        [ax.invert_yaxis() for ax in axes]
        axes[0].set_title('A.Ninwavelet Morse (sigma=7.0)')
        axes[1].set_title('B.Ninwavelet Morlet (beta=17.5, gamma=3)')
        axes[2].set_title('C.MNE Morlet (n_cycles=7.0)')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        axcbs = [make_axes_locatable(ax).new_horizontal(size='2%', pad=0.05)
                 for ax in axes]
        [fig.add_axes(ac) for ac in axcbs]
        [ax.set_ylabel('Frequency (Hz)', fontsize=24) for ax in axes]
        [ax.set_xlabel('Time (Second)', fontsize=24) for ax in axes]
        [plt.colorbar(im, cax=cb) for im, cb in zip(ims, axcbs)]
        [ax.set_aspect('auto') for ax in axes]
        [ax.set_yticks(np.arange(0, 70, 20)) for ax in axes]
        [ax.set_yticklabels(np.arange(30, 100, 20), fontsize=24)
         for ax in axes]
        [ax.set_xticks(np.arange(0, 1000, 200)) for ax in axes]
        [ax.set_xticklabels(np.arange(0, 1000, 200), fontsize=24)
         for ax in axes]
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0.4, hspace=0.5)
        fig.savefig('/home/ninja/Frontiers_LaTex_Templates/imgs/cwt.jpg')
        plt.show()
    print(f'Morse max if {np.max(np.abs(result_morse))}')
    print(f'Morlet max is {np.max(np.abs(result_morlet))}')
    print(f'MNE max is {np.max(np.abs(result_mne))}')
    print(f'Morse mean is {np.abs(result_morse).mean()}')
    print(f'Morlet mean is {np.abs(result_morlet).mean()}')
    print(f'MNE mean is {np.abs(result_mne).mean()}')
    # result_morse = morse.power(sin, reuse=True)
    plot_tf(np.abs(result_morse[0] - result_morlet[0]))
    # plt.show()


def other_wavelet_test() -> None:
    hz = 30, 31
    s = 7
    shannon = Shannon().make_wavelets(hz)
    morlet = Morlet(sigma=s).make_wavelets(hz)
    plt.plot(shannon[0])
    plt.plot(morlet[0])
    plt.show()
    plt.plot(Morlet().make_fft_wavelets(hz)[0])
    plt.plot(Shannon().make_fft_wavelets(hz)[0])
    plt.show()
    # sin = make_example(1, False)

    # log.info('Other wavelets')
    # log.info('Haar wavelets')
    # result_haar = Haar(1000).power(sin, np.arange(1, 1000, 1))
    # plot_tf(result_haar)

    # log.info('Other wavelets')
    # log.info('Mexican wavelets')
    # result_mexican = MexicanHat(1000).power(sin, np.arange(1, 1000, 1))
    # plot_tf(result_mexican)


def fft_wavelet_test() -> None:
    hz = 10.
    r = 3
    b = 17.5
    s = 7
    morse = Morse(r=r, b=b)
    nin_morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet.mode = CWTMode.Normal
    w = morse.make_wavelets([hz])[0]
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
    ax.plot(normal_morlet.make_wavelets([hz])[
            0], label='Morlet Wavelet Normal Mode')
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
    from mne.io import Raw
    raw = Raw('/home/ninja/ninja.fif')
    data = raw.get_data()[raw.ch_names.index('EEG O1-Ref')]
    d = data[150*500: 190*500]
    d = Baseline(d, 1000, 0, 1).zscore()
    tf = Morse(raw.info['sfreq'], cuda=True).power(d, np.arange(0.1, 50, 0.1))
    ax = plot_tf(tf, frange=(0, 50, 10),
                 trange=(0, 40, 10), sfreq=500, show=False)
    ax.set_title('My EEG power(O1)')
    ax.set_xlabel('Time Course(sec)')
    ax.set_ylabel('Hz')
    plt.show()


def speed_wavelet_test() -> None:
    from mne.time_frequency.tfr import morlet, cwt
    freq_range = 30, 500
    freqs = np.arange(freq_range[0], freq_range[1], 1)
    t = time()
    for n in range(10):
        mne_morlet = morlet(1000, freqs)
    print('MNE wavelet generation: ', time() - t)

    t = time()
    for n in range(10):
        nin_morlet = Morlet(1000, gabor=True).make_fft_wavelets(freqs)
    print('Nin wavelet generation: ', time() - t)


def speed_test(repeat: int) -> None:
    length = 2
    repeat_s = repeat
    freq_range = 30, 500
    t = time()
    sin = make_example(length, False, False)
    # c_sin = make_example(length, False, False)
    check = False
    wv = make_example(length, False, False)
    wv_mne = np.array([make_example(length, False, False)])
    freqs = np.arange(freq_range[0], freq_range[1], 1)
    c_freqs = cp.arange(freq_range[0], freq_range[1], 1)
    times: List[float] = []
    labels: List[str] = []

    # Warm up
    nin_morlet = Morlet(cuda=True, sfreq=1000, cache_limit=0)
    result_morlet = nin_morlet.power(cp.asarray(sin), c_freqs)
    # ====================
    # Scipy
    # ====================
    from scipy import signal
    labels.append('Scipy morlet')
    t = time()
    for n in range(repeat_s):
        result_sci = np.abs(signal.cwt(wv, signal.morlet2, freqs)) ** 2
    times.append(time() - t)

    # ====================
    # MNE
    # ====================
    from mne.time_frequency.tfr import morlet, cwt
    labels.append('MNE morlet')
    t = time()
    mne_morlet = morlet(1000, freqs)
    for n in range(repeat_s):
        result_mne = np.square(np.abs(cwt(wv_mne, mne_morlet)))
    times.append(time() - t)

    # ====================
    # SWAN
    # ====================
    from swan import pycwt
    swan_morlet = pycwt.Morlet()
    t = time()
    for n in range(repeat_s):
        r = np.abs(pycwt.cwt_f(wv, freqs, 1000, swan_morlet)) ** 2
    times.append(time() - t)
    labels.append('Swan')

    # ====================
    # NinWavelet
    # ====================

    # nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=0)
    # nin_morlet.mode = CWTMode.Normal
    # t = time()
    # for n in range(repeat):
    #     result_morlet = nin_morlet.power(sin, freqs)
    # print(result_morlet[-1, -1])
    # times.append(time() - t)
    # labels.append('Ninwavelets\nCPU No cache normal')

    # nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=0)
    # t = time()
    # for n in range(repeat):
    #     result_morlet = nin_morlet.power(sin, freqs)
    # print(result_morlet[-1, -1])
    # times.append(time() - t)
    # labels.append('Ninwavelets\nCPU No cache')

    nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=1)
    t = time()
    for n in range(repeat):
        result_morlet = nin_morlet.power(sin, freqs)
    print(result_morlet[-1, -1])
    times.append(time() - t)
    labels.append('Ninwavelets\nSingle-core CPU')

    nin_morlet = Morlet(cuda=False, sfreq=1000, cache_limit=1)

    def wrap(func, q, *args):
        q[:] = func(*args)[:]
    t = time()
    for n in range(repeat):
        bufs = [np.empty((freqs.shape[0], sin.shape[0]))]

        ps = [Thread(target=wrap, args=(nin_morlet.power, buf, sin, freqs))
              for buf in bufs]
        [p.start() for p in ps]
        [p.join() for p in ps]
    # for n in range(repeat):
    #     result_morlet = nin_morlet.power(sin, freqs)
    print(bufs[0][-1, -1])
    times.append(time() - t)
    labels.append('Ninwavelets\nMulti-core CPU')

    # nin_morlet = Morlet(cuda=True, sfreq=1000, cache_limit=0)
    # nin_morlet.mode = CWTMode.Normal
    # t = time()
    # for n in range(repeat):
    #     result_morlet = nin_morlet.power(cp.asarray(sin), c_freqs)
    # print(cp2np([result_morlet[-1, -1]], np.float))
    # times.append(time() - t)
    # labels.append('Ninwavelets\nNormal')

    # nin_morlet = Morlet(cuda=True, sfreq=1000, cache_limit=0)
    # t = time()
    # for n in range(repeat):
    #     result_morlet = nin_morlet.power(cp.asarray(sin), c_freqs)
    # print(cp2np([result_morlet[-1, -1]], np.float))
    # times.append(time() - t)
    # labels.append('Ninwavelets\nFrom Fourier')

    t = time()
    nin_morlet = Morlet(cuda=True, sfreq=1000, cache_limit=1)
    # for n in range(repeat):
    result_morlet = [cp.asnumpy(nin_morlet.power(
        cp.asarray(sin), c_freqs)) for n in range(repeat)]
    # result = result_morlet.get()
    # result = list(cp2np(result_morlet, np.float))[0]
    times.append(time() - t)
    labels.append('Ninwavelets\nGPU for each wave')
    print(times)
    print(labels)

    sp_c_sin = cp.array([sin for n in range(10)])
    nin_morlet = Morlet(cuda=True, sfreq=1000, cache_limit=1)
    labels.append('Ninwavelets\nby GPU grid')
    t = time()
    for n in sp_c_sin // 10:
        result_morlet_cuda = nin_morlet.power(sp_c_sin, c_freqs)
    print(result_morlet_cuda[-1, -1, -1])
    times.append(time() - t)

    bars = np.arange(0, len(times), 1)
    plt.bar(bars / bars[1], 1/np.array(times))
    plt.xticks(bars / bars[1], labels, fontproperties=font_prop)
    # plt.xlabel('Packages')
    plt.ylabel(f'Speed. ({repeat}trial / sec) Bigger is fast.')
    plt.title(f'{length}sec wave, Sampling frequency'
              f'1000Hz\nMorletWavelet({freq_range[0]}~{freq_range[1]}Hz)'
              f'{repeat}times')
    plt.show()


def geom_test() -> None:
    morse = Morse()
    result = morse.power(np.random.random(1000), np.geomspace(1, 10, 1000))
    plt.imshow(result)
    plt.show()


def tune(wave: Array) -> None:
    morse = Morse()
    for n in range(100):
        result = morse.power(wave, freqs)


def tune_cuda(wave: Array, repeat: Optional[int] = None) -> None:
    ncp = cp if cuda else np
    gabor = False
    morlet = Morlet(sfreq=2048, real_wave_length=2,
                    cuda=False, cache_limit=0, gabor=gabor)
    morlet_cp = Morlet(sfreq=2048, real_wave_length=2,
                       cuda=True, cache_limit=0, gabor=gabor)
    repeat = 100 if repeat is None else repeat
    freq = np.arange(1, 100, 1)
    freq_cp = cp.arange(1, 100, 1)

    morlet.cache_limit = 0
    t = time()
    for n in range(repeat):
        result = morlet.make_fft_wavelets(freq)
        i = result[0, 0]
    print('Morlet', (time() - t))

    morlet.mode = CWTMode.Normal
    t = time()
    for n in range(repeat):
        result = morlet.make_fft_wavelets(freq)
        i = result[0, 0]
    print('Morlet Normal', (time() - t))

    t = time()
    for n in range(repeat):
        result = morlet_cp.make_fft_wavelets(freq_cp)
        i = result[0, 0]
    print('Morlet GPU', (time() - t))

    morlet.mode = CWTMode.Normal
    t = time()
    for n in range(repeat):
        result = morlet_cp.make_fft_wavelets(freq_cp)
        i = result[0, 0]
    print('Morlet Normal GPU', (time() - t))

    morse = Morse(sfreq=2048, real_wave_length=2,
                  cuda=False, cache_limit=0)
    t = time()
    for n in range(repeat):
        result = morse.make_fft_wavelets(freq)
        i = result[0, 0]
    print('Morse', (time() - t))

    morse = Morse(sfreq=2048, real_wave_length=2,
                  cuda=True, cache_limit=0)
    t = time()
    for n in range(repeat):
        result = morse.make_fft_wavelets(freq_cp)
        i = result[0, 0]
    print('Morse GPU', (time() - t))


def tune_2d(wave: Array) -> None:
    morse = Morse(cuda=True)
    result = morse.power(wave, freqs)


def test_2d() -> None:
    morse = Morse()
    wv = cp.array([make_example(1, True) for n in range(2)])
    freqs = np.arange(30, 50, 1)
    morse.cwt(wv, freqs)
    morse.cwt(wv[0], freqs)


def pad() -> None:
    morlet = Morlet(1000)
    wv = make_example(1.0015, cuda=True)
    freqs = cp.arange(30, 500, 1)
    t = time()
    for n in range(100):
        x = morlet.cwt(wv, freqs, padding=True)
    print(x.shape)
    print(time() - t)
    morlet = Morlet(1000)
    t = time()
    for n in range(100):
        x = morlet.cwt(wv, freqs)
    print(x.shape)
    print(time() - t)


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
        # test3d()
        # fft_wavelet_test()
        other_wavelet_test()
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
            repeat = int(argv[2]) if argv[2] else 1
            speed_test(repeat)
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
        tune_cuda(cp_wave, int(argv[2]))
        # tune_2d(sp_cp_wave)
    if 'wavelet_speed' in argv:
        speed_wavelet_test()
    if 'window' in argv:
        wave = np.ones(1000)
        plt.plot(windows.hanning_window(
            1000, cuda=False, tukey_ratio=0.1) * wave)
        plt.plot(np.abs(ifft(wave)))
        plt.show()
    if 'app' in argv:
        sin = make_example(1, False, False)
        sin10 = np.tile(sin, 100).reshape((100, sin.shape[-1]))
        morse = Morse()
        freqs = np.arange(30, 150)
        res = morse.cwt(sin, freqs)
        t = time()
        for n in range(100):
            res = morse.fourier_cwt(fft(sin), freqs)
        print(time() - t)
        plt.imshow(np.abs(res))
        plt.show()
        morse1 = experimental.Morse()
        t = time()
        res = morse1.app_cwt(sin10, freqs, band_rate=0.1)
        print(time() - t)
        plt.imshow(np.abs(res[50]))
        plt.show()
    if 'pad' in argv:
        pad()
    if 'pres' in argv:
        precision()

    t = time()
    # i = cp2np([cp.arange(3, 19) for n in range(4000)])
    # print(i[0])
    # print(time() - t)
    # t = time()
    # i = [cp.arange(3, 19) for n in range(4000)]
    # print(i[0])
    # print(time() - t)
