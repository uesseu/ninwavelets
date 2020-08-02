# NinWavelets
This is a python package for analystic wavelet transform.  
Morlet, Shannon, Generalized Morse(GMW) and so on.  
It is based on Numpy or Cuda, and it may be fast especially on Cuda.  

![My EEG Power!](img/alpha.png)  
This is my alpha band of EEG which was processed by this package.

# Why NinWavelets?
There may be some advantages and critical(?) limitations.  
Please see Advantages and Limitations.  

- Use wavelets which is originally Frourier transformed
    + Generalized Morse(I wanted this and so I wrote it.)
    + Morlet/Gabor(Frourier transformed version.)
    + Shannon(It looks like Haar, when fourier transformed.)
    + May be more(It is easy to scale!)
- Skipping one FFT when performing CWT.
    + May be better and faster if you use FFT method.
- Cuda
    + If you want to process long wave, it may be extremely faster!
- Compatibility
    + You can use it with mne-python.(Sensor based only...)
    + Numpy and Cupy is available.
- Reliability(???)
    + I am not a scholar, not a engineer, just a nurd.
    + Just read the source code! It is obligation for freedom if you use OSS.
- In heavyly debelopment
    + Destructive change may performed.

# Install

```bash
pip install git+https://github.com/uesseu/ninwavelets
```

# Dependency
- python 3.6.5 or newer(It involves type hint and annotation)

These are automatically installed.  
- scipy
- numpy

If you want to use cuda, please setup cuda and cupy.  
[https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)  
[https://www.geforce.com/drivers](https://www.geforce.com/drivers)  

and then, run

```bash
pip install cupy
```

It is faster if you process long wave, like my EEG power example.  

Optionally, if you want to process EEG/MEG, you can use this.  

- mne

```bash
pip install mne
```

# Purpose and background
At first, this package was written to perform GMW with mne python.  
But I found that, using mne function with this package is ugly way.  
Because GMW needs inverse Fourier transform to no purpose.  
GMW should skip one inverse FFT.(Skipping is beautiful)  
Now it has own CWT method, which can skip one FFT.  
And I noticed, it is good for Morlet Wavelet too.  


It is a brand new project, and under heavily development(On my Sunday).  
Destructive changes may be made.  
I do not think, you want to use it, but when you use it, check the version strictly!  


# Exsamples
GMW is similar to morlet wavelet, if you use default param.  

You can calculate complex value and power.  

```python
from ninwavelets import Morse

morse = Morse(1000, gamma=3, beta=17.5)
freq = 60
time = np.arange(0, 0.3, 0.001)

# Now lets analyze sin wave!
sin = np.array([np.sin(time * freq * 2 * np.pi)])

result = morse.power(sin, range(1, 100))
complex_result = morse.cwt(sin, range(1, 100))
plt.imshow(result, cmap='RdBu_r')
plt.gca().invert_yaxis()
plt.title('CWT of 60Hz sin wave')
plt.show()
```


You can perform baseline correction.

```python
from ninwavelets import Baseline

wave = Baseline(wave, 1000, 0, 0.2).zscore()
```

You can also use plot_tf().
It can plot result with colorbar.

```python
from ninwavelets import plot_tf
plot_tf(result)
```

If you just want to perform cwt only, write like this.  

```python
result = morse.cwt(sin, range(1, 100))
```

If you are mne user, epochs can be processed.  
But it is not so useful. See 'NinWavelets for MNE'.  

These are results from my test code.  

![various](img/various.png)

![cwt](img/cwt.png)

# Mode of wavelets
This package has some mode to calculate CWT.  
You can use them like this.  

```
from ninwavelet import Morlet, CMTMode
morlet = Morlet().set_mode(CMTMode.Normal)
```

set_mode method retuns it self ;).

|Name|nature|
|--|--|
|Normal|Easy to understand. May be precise for Morlet.|
|Fast|Very fast, may be best for GMW and Morlet|
|Convolve|Just slow. Sloooooooow!|


Wrighting in english may not be easy to understand.  
Formula is like below.

### Normal mode
When Normal mode, it uses fft to calculate fast.

```
InverseFFT(Convolve(FFT(wavelet) @ FFT(wave_to_analyze)))
```

### Fast mode
When Fast mode, it uses fourier transform version of formula.  
Default for Morlet and GMW.  

```
InverseFFT(Convolve(Transformed_wavelet @ FFT(wave_to_analyze)))
```

### Convolve mode
When Convolve, it yields raw wave by inverse fourier transform.  
Default for Haar.

```
Convolve(Wavelet @ wave_to_analyze)
```

But, there is some wavelets, which has no formula of raw wave.  
For example, GMW is...  

```
GMW = InverseFFT(raw_formula)
```

In this case, it perform inverse fft before convolving.  

Normal is easy to understand, but slower than Fast.  
Fast is ofcource fast, more precise than Normal, and can use cuda.  
Convolve may too heavy and not good method for GMW but it may be good for Haar.

If there is a formula of fourier transformed wavelet,  
Reverse mode may be best mode.  
But, not all the wavelet has formula.  
For example, I do not know the formula of raw GMW.  
(Generally, GMW is calculated by inverse fourier transform.)

This pseudo code let ninwavelet calculate by raw wave of morse.

```python
from ninmne import CWTMode, Morlet
morlet = Morlet(1000)
morlet.mode = WaveletMode.Normal
transformed = morlet.power(wave)
```

# Reference
## WaveletClasses

They are classes for wavelet. They are sub class of WaveletBase.  
You can inherit 'WaveletBase' class and make your own wavelet.  
I wrote some wavelets.  

For example, lets see Morse class!  


```python
from ninwavelets import Morse
```

```python
Morse(self, sfreq: float = 1000,
      b: float = 17.5, r: float = 3,
      length: float = 10, cuda: bool = False) -> None:
```

Parameters

| Param       | Type  | Default |                                                              |
| --          | --    | --      | --                                                           |
| sfreq       | float | 1000Hz  | Sampling frequency.                                          |
| b           | float | 17.5    | beta value                                                   |
| r           | float | 3       | gamma value. 3 may be good value.                            |
| length      | float | 10      | Length paramater. It affects only when you plot wavelets.    |
| cuda        | bool  | False   | Use cuda or not. See 'Performance of wavelet transform'.     |

```python
morse = Morse()
```

But, I dont know whether it is good or bad...  

This is an example. List of wavelet classes is this.  
All of them are in module 'nin_wavelets.wavelets' and
you can use it by this code.

```python
from nin_wavelets import hoge
```

| Name              | Name in this package              |
|-------------------|-----------------------------------|
| Generalized Morse | Morse                             |
| Complex Morlet    | Morlet                            |
| Complex Shannon   | Shannon                           |
| Gausian(Gabor)    | Morlet(gabor option is available) |
| MexicanHat        | MexicanHat                        |
| Haar              | Haar(This is not good!)           |

### make_wavelets

Exsample.

```python
wavelet = Morse(1000, 17.5, 3).make_wavelets([10])[0]
```

Make list of wavelets.  

| Param | Type  |                      |
|-------|-------|----------------------|
| freq  | float | List of frequencies. |

Because it returnes bad wave easily,  
you should use it when you plot it only.  
For example, GMW with sfreq=1000, freq=3 returnes bad wave.  
If you want good wave, you must set  

Returns  
MorseWavelet: list of np.ndarray  

### make_fft_waves

```python
make_fft_waves(self, total: float, one: float,
               freqs: Iterable) -> Iterator:
```
Make Fourier transformed Wavelet.
If the wavelet is originally Frourier transformed wavelet,
it just calculate original formula.
If wavelet is originally not Fourier transformed wavelet,
it run FFT to make them.

### cwt
CWT method.

| Param    | Type  |                                                                      |
|----------|-------|----------------------------------------------------------------------|
| wave     | float | Wave drawed by numpy.                                                |
| freqs    | float | List of frequencies.                                                 |
| max_freq | float | Max freq.                                                            |
| reuse    | bool  | Reuse wavelets you made before. If true, calculation becomes faster. |

```python
def cwt(self, wave: np.ndarray,
        freqs: Union[List[float], range, np.ndarray],
        max_freq: int = 0, reuse=True) -> np.ndarray:
```

example

```python
import numpy as np
freq: float = 60
length: float = 5

time: np.ndarray = np.arange(0, length, 0.001)
sin = np.array(np.sin(time * freq * 2 * np.pi))
morse = Morse()
result = morse.cwt(sin, np.arange(1, 1000, 1))
plt.imshow(np.abs(result), cmap='RdBu_r')
plt.show()
```

max_freq is a param to cut result.

## power
```
power(self, wave: np.ndarray,
      freqs: Union[List[float], range, np.ndarray],
      reuse=True) -> np.ndarray:
```

Run cwt of mne-python, and compute power.

| Param    | Type  |                                                                      |
|----------|-------|----------------------------------------------------------------------|
| wave     | float | Wave drawed by numpy.                                                |
| freqs    | float | List of frequencies.                                                 |
| max_freq | float | Max freq.                                                            |
| reuse    | bool  | Reuse wavelets you made before. If true, calculation becomes faster. |

Returns  
Result of cwt. np.ndarray.  


## MorseMNE Class(Bad way)

MorseMNE class to use function of MNE-python,  
which is Great package to analyze EEG/MEG.  
It is same as Morse class except cwt but  
if you run cwt, it uses mne.time_frequency.tfr.cwt to run cwt.  

But it is not recommended, because mne.time_frequency.tfr.cwt needs  
wavelet which is 'not Fourier transformed'.  
Basically, GMW is a wavelet which is originally  
'Fourier transformed wavelet' and so, you need to run  
InverseFourier transform before you perform CWT.  
I think, this ugly class is disgusting.  

By the way, there is a formula of Morlet wavelet which is Fourier transformed.  
And so, I think, it may be better to use the formula  
even if you use Morlet Wavelet.  

## Baseline Class

NinWavelets supports baseline correction.

```python
def __init__(self, wave: Array, sfreq: float,
             start: float, stop: float) -> None:
```

You need to import Baseline.
In this case, wave was read as 'wave'.

```python
from ninwavelets import Baseline

baseline = Baseline(wave, 1000, 0, 0.2)
wave = baseline.zscore()
```

In this case, 1000 is sampling frequency.
0 ~ 0.2 second is range for baseline correction.

There are these methods.

- mean: subtraction
- ratio: division
- percent: division after subtraction
- log: log after division
- log10: log10 after division
- zscore: standize after subtraction
- zlog: log after zscore



## NinWavelets for MNE

ninwavelets.EpochsWavelet is a class for Epochs class of mne.

```python
from ninwavelets import EpochsWavelet, Morse, plot_tf
from mne import read_epochs

fname = 'hoge_epo.fif'
epochs = read_epochs(fname)
morse = Morse()
result = EpochsWavelet(epochs, morse).power(range(1, 100))
plot_tf(result)
```

At first, make instance of wavelets(Morse, Morlet and so on).
Then, make EpochsWavelet class.
This has methods named cwt, power and itc.
plot_tf is a function to plot numpy array.

## WaveletBase Class
Super class of wavelets.
You can inherit this class and make new wavelets.

After inherit this, you can edit these methods.  

- BaseWavelet.formula
- BaseWavelet.trans_formula
- BaseWavelet.peak_freq
- BaseWavelet.cp_formula
- BaseWavelet.cp_trans_formula

At first, you need to overwrite them.  
They needs to written by numpy or cupy.  
Cupy version should start with 'cp'.  
These methods are used in the class, and bothering procedures are done.

## Way to inherit

This is an example.
This code is sub class of BaseWavelet, and is
ninwavelets.MorletWavelet.

```python
import cupy as cp
from ninwavelet import WaveletBase
import matplotlib.pyplot as plt


class Morlet(WaveletBase):
    '''
    Morlet Wavelets.
    Example.
    >>> morse = Morse(1000, sigma=7.)
    >>> freq = 60
    >>> time = np.arange(0, 0.3, 0.001)
    >>> sin = np.array([np.sin(time * freq * 2 * np.pi)])
    >>> result = morse.power(sin, range(1, 100))
    >>> plt.imshow(result, cmap='RdBu_r')
    >>> plt.gca().invert_yaxis()
    >>> plt.title('CWT of 60Hz sin wave')
    >>> plt.show()

    Parameters
    ----------
    sfreq: float | Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float | sigma value
    length: float | Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
        Peak wave * length is the length of wavelet.

    Returns
    -------
    As constructor, Morse instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 real_wave_length: float = 1.,
                 gabor: bool = False, cuda: bool = False) -> None:
        super(Morlet, self).__init__(sfreq, real_wave_length, cuda)
        self.mode = CWTMode.Fast
        self.sigma = sigma
        self.c = np.float_power(1 +
                                np.exp(-np.float_power(self.sigma, 2) / 2)
                                - 2 * np.exp(-3 / 4
                                             * np.float_power(self.sigma, 2)),
                                -1/2)
        self.k = 0 if gabor else np.exp(-np.float_power(self.sigma, 2) / 2)

    def cp_trans_formula(self, freqs: cp.ndarray,
                                 freq: float = 1.) -> cp.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        result = (self.c * cp.pi ** (-1/4) *
                  (cp.exp(-cp.square(self.sigma-freqs) / 2) -
                   self.k * cp.exp(-cp.square(freqs) / 2)))
        return result

    def trans_formula(self, freqs: np.ndarray,
                              freq: float = 1) -> np.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        return (self.c * np.float_power(np.pi, (-1/4)) *
                (np.exp(-np.square(self.sigma-freqs) / 2) -
                 self.k * np.exp(-np.square(freqs) / 2)))

    def formula(self, timeline: np.ndarray,
                        freq: float = 1) -> np.ndarray:
        return (self.c * np.float_power(np.pi, (-1 / 4))
                * np.exp(-np.square(timeline) / 2)
                * (np.exp(self.sigma * 1j * timeline) - self.k))

    def cp_formula(self, timeline: cp.ndarray,
                        freq: float = 1) -> cp.ndarray:
        return (self.c * (cp.pi ** (-1 / 4))
                * cp.exp(-cp.square(timeline) / 2)
                * (cp.e ** (self.sigma * 1j * timeline) - self.k))

    def peak_freq(self, freq: float) -> float:
        return self.sigma / (1. - np.exp(-self.sigma * freq))
```

All you should do is write formula.  
The formulas may be written in mathmatical papers! ;)  

## Perfomance of wavelet transform
Optionally, you can write code for cupy.  
If you want to use cupy, write cp version and name it like below.

- self.cp_trans_formula
- self.cp_formula
  
From version 0.0.3, It became fast.  

I performed benchmark test by my NotePC  
'Dell G3 15-3579r' with Intel corei7(4.1Ghz 6core) and Geforce GTX1050.  

version 0.0.2
One morse wavelet transform
Sampling freq: 1000
| Length | back ground | CWT time |
|--------|-------------|----------|
| 1sec   | cupy        | 1.28sec  |
| 1sec   | numpy       | 0.872sec |
| 50sec  | cupy        | 7.25sec  |
| 50sec  | numpy       | 15.9sec  |

version 0.1.0
50 morse wavelet transform
Sampling freq: 1000
| Length | back ground | CWT time |
|--------|-------------|----------|
| 1sec   | cupy        | 1.39sec  |
| 1sec   | numpy       | 1.29sec  |
| 50sec  | cupy        | 2.04sec  |
| 50sec  | numpy       | 72.6sec  |


# Advantages and Limitations

## Method

Some mathmatician say, DFT is not good way.  
We have no good method to perform Fourier transform by digital computer.  
Method of convolve may be good way for Wavelet transform.  
But GMW needs Frourier transform.  
Further more, convolving needs long long loooong time.  
And so, convolving method is useless.  
It is not good for Morlet wavelet too. There may be some methods.  

**1**

```
Convolve(wave, wavelet)  # Very good? But slooooooooooow!
```

**2**

```
iFFT(FFT(wave) * FFT(wavelet))  # Fast, and widely used. But not good. 
```

**3**

```
iFFT(FFT(wave) * FFTed_wavelet)  # Better and faster than 2.
```

I adopted method 3. Not only GMW, but also Morlet wavelet will be performed by 3.

## It is just my hobby(Critical limitation!)
This project is just my hobby, and I am not an engineer or scholar, just a nurd.  
I was said, "It is impossible for you to write reliable code of Wavelet Transform".  
And so, there must be lots of bugs and IT IS USELESS.  
**You should not use this package for work!**  
(In fact, this package is useful "for me".)  

~~Ofcource, I can not say "This is reliable code" at all.~~  
~~But is it impossible to write python for non engineer?~~  
~~Is it impossibe to understand wavelet transform for non scholar?~~  

# Contribution
I am glad to receive contribution!  
Because I am a lonely nurd, if I could speak about wavelet, I am pleased.  
I want to hear constructive oppinions, if I could.  

But, I cannot receive such contribution like...  

- Delete whole of this repository to delete bugs!
- I am a PHD, and so, just change the authors name and make it reliable!

I want to enjoy, and if I could, I want to let you enjoy.  
It is the rule of this project.  

# Licence
Copyright (c) 2020 Forest Segne
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# TODO

- Other wavelets
    + [x] Morse
        * [x] Compared with mne
    + [x] Morlet
        * [x] Compared with mne
    + [x] Gabor
        * [x] Compared with mne
    + [x] Mexican hat
    + [x] Shannon
    + [x] Haar
    + [x] Scalability for unknown wavelets
- More methods
    + [ ] Decimation
    + [ ] DWT
    + [ ] 2D wavelet
- [x] Use cuda or cython and speedup!
    + [x] It was cythonized before. But it is not good for scalability.
    + [x] It may be faster with cupy if you process long wave.
    + [x] Now, It is extremely fast!
- [ ] Kill typos(I am a Nip and not good at English) ;(
- [x] Licence
    + [x] Whether write my name or not.
        * [x] I wrote one of my handle name.
    + [x] Which licence to use.
        * [x] MIT licence.
- [ ] CUPY related limitation
    + [ ] convolve is not allowed in cupy 7.6
        * [ ] I am waiting, but it seems to be needless.
