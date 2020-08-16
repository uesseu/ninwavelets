# NinWavelets
This is a python package for analystic wavelet transform.  
Morlet, Shannon, Generalized Morse(GMW) and so on.  
It is based on Numpy or Cuda, and it may be fast especially on Cuda.  

![My EEG Power!](img/alpha.png)  
This is my alpha band of EEG which was processed by this package.

```python
import cupy as cp
from ninwavelets import Morse
freqs = cp.array(20, 100, 1)

wave = cp.sin(cp.arange(0, 1000, 1))
morse = Morse(1000)
morse.cwt(wave, freqs)
# morse.cwt(wave, cp.arange(20, 100, 1))   is slow!
```


# Why NinWavelets?
There may be big advantages and limitations.  
Please see Advantages and Limitations.  

- Use wavelets which is originally Frourier transformed
    + Generalized Morse(A flexible wavelet, which has two parameters.)
    + Morlet/Gabor(Frourier transformed version of Morlet/Gabor.)
    + Shannon(It looks like Haar, when fourier transformed.)
    + May be more(It is easy to scale!)
- Skipping one FFT when performing CWT.
    + May be better and faster if you use FFT method.
- Speed
    + When you use cuda and process longwave, may be extremely faster!
    + Even if you use numpy, it is very very fast!
- Reliability
    + Being brand new project, it has no achivement...
    + There may be lots of bugs.
    + Do not rely on it! Read the source code! Test by your self!
- In heavily debelopment
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

Ninwavelets with cuda is extremely fast if you process long wave,  
like my EEG power example.  


# Usage
- At first, import wavelets and import numpy or cupy.
- Ninwavelets can use cuda, and so, switch numpy or cupy.
- The wave must be numpy or cupy data.
  + If you want to use cuda, prepare cupy data! It does not transform!
- Make instance of wavelet
- Make frequency instance as numpy, cupy or range.
  + If you did not make it, ninwavelets may becomes extremely slower!
- Perform cwt.


```python
import cupy as cp
from ninwavelets import Morse
freqs = cp.array(20, 100, 1)

wave = cp.sin(cp.arange(0, 1000, 1))
morse = Morse(1000)
morse.cwt(wave, freqs)
# morse.cwt(wave, cp.arange(20, 100, 1))   is slow!
```

# Purpose and background
At first, this package was written to perform GMW on mne python.  
But I wrote CWT. Because GMW can skip one inverse FFT.  
Now it has own CWT method, which can skip one inverse FFT.  
And I noticed, it is good for Morlet Wavelet too.  
The result resembles that of mne python.  

This is a brand new project, and under heavily development(Mainly on my Sunday).  
Destructive changes may be made.  

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

# Modes of wavelets
This package has some modes to calculate CWT.  
You can use them like this.  

```
from ninwavelet import Morlet, CMTMode
morlet = Morlet().set_mode(CMTMode.Normal)
```

set_mode method retuns it self ;).

| Name     | nature                                         |
|----------|------------------------------------------------|
| Normal   | Easy to understand. May be precise for Morlet. |
| Fast     | Very fast, may be best for GMW and Morlet.     |
| Convolve | Just slow. Sloooooooow!                        |
| Reverse  | Do not use this.                               |


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

```
Convolve(InverseFFT(Transformed_wavelet) @ wave_to_analyze)
```

It seems bad, and this is why I wrote this package.  

### Reverse mode
Even if there is formula of wavelet, in this mode,  
ninwavelet try to perform inverseFFT before convolving.  


## About modes
Normal mode is easy to understand, can use cuda, but a little slower than Fast mode.  
Fast mode is fast, may be more precise than Normal, and can use cuda.  
Convolve mode may too heavy and not good method for GMW but it may be most precise for Morlet.
Reverse mode is ugly way and just for debugging.  

Not all the wavelets have formula.  
For example, I do not know the formula of raw GMW.  
(Generally, GMW is calculated by inverse fourier transform.)

This pseudo code let ninwavelet calculate by iFFTed wave of morse.

```python
from ninmne import CWTMode, Morlet
morlet = Morlet(1000)
morlet.set_mode(WaveletMode.Convolve)
transformed = morlet.power(wave)
```

CWTMode should be set properly when the class is inherited.  

# Reference
## WaveletClasses

They are classes for wavelet. They are sub classes of WaveletBase.  
You can inherit 'WaveletBase' class and make your own wavelet.  
I wrote some wavelets.  

For example, lets see Morse class!  


```python
from ninwavelets import Morse
```

```python
Morse(self, sfreq: float = 1000,
      b: float = 17.5, r: float = 3,
      real_wave_length: float = 10, cuda: bool = False) -> None:
```

Parameters

| Param            | Type  | Default |                                                          |
| --               | --    | --      | --                                                       |
| sfreq            | float | 1000Hz  | Sampling frequency.                                      |
| b                | float | 17.5    | beta value                                               |
| r                | float | 3       | gamma value. 3 may be good value.                        |
| real_wave_length | float | 10      | Length of wavelet(sec). It is modified when CWT.         |
| cuda             | bool  | False   | Use cuda or not. See 'Performance of wavelet transform'. |

```python
morse = Morse(1000)
```

List of wavelet classes is written bellow.  
All of them are in module 'nin_wavelets.wavelets' and you can use it by such code.  

```python
from nin_wavelets import Morse
```

| Name              | Name in this package                   |
|-------------------|----------------------------------------|
| Generalized Morse | Morse                                  |
| Complex Morlet    | Morlet                                 |
| Complex Shannon   | Shannon                                |
| Gausian(Gabor)    | Morlet(gabor option is available)      |
| MexicanHat        | MexicanHat                             |
| Haar              | Haar(This is not good in ninwavelets!) |

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
When you perform CWT, ninwavelets can select best way,  
and the best way do not use raw wavelet in some cases.  
For example, GMW with sfreq=1000, freq=3 returnes bad wave.  
If you want good wave, you must set 'real_wave_length' at constructor.  
When it performs CWT, real_wave_length will be set properly.  

Returns  
MorseWavelet: list of np.ndarray  

### make_fft_waves

```python
make_fft_waves(self, total: float, one: float,
               freqs: Iterable) -> Iterator:
```

Make Fourier transformed Wavelet.  
If the wavelet is originally Frourier transformed wavelet  
and it has 'fft_formula', it just calculate original formula.  
If wavelet is originally not Fourier transformed wavelet,  
it run FFT to make them.  

### cwt
Perform CWT.  

Parameters
| Arg    | Type                          |                        |
|--------|-------------------------------|------------------------|
| wave   | Union[np.ndarray, cp.ndarray] | Raw wave to transform. |
| freqs  | List[float]                   | Frequencies.           |
| reuse  | bool = True                   | Reuse wavelet or not.  |
| logger | Logger = logger               | Custom logger          |

If raw wave length differ from length of wavelet,  
or freqs are differ from previous transformation,  
recreate wavelet even if "reuse" is True.  

Returns result of CWT, which type is Union[np.ndarray, cp.ndarray].  


Example

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


## power
Same as CWT, but returns power value.  
Power value is square(abs(cwt)).

## power
Same as CWT, but returns absolute value.  

## Baseline Class

NinWavelets supports baseline correction.  
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



## WaveletBase Class
Super class of wavelets.  
You can inherit this class and make new wavelets.  

After inherit this, you should edit these methods.  

- BaseWavelet.formula
- BaseWavelet.trans_formula
- BaseWavelet.peak_freq
- BaseWavelet.cp_formula
- BaseWavelet.cp_trans_formula

At first, you need to overwrite them.  
They needs to be written by numpy or cupy code.  
Cupy version should start with 'cp'.  
By inheriting, ninwavelet becomes scalable.  

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

# Advantages and Limitations

## Speed
From version 0.0.3, It became extremely fast.  

I performed benchmark test by my NotePC  
'Dell G3 15-3579r' with Intel corei7(4.1Ghz 6core) and Geforce GTX1050.  
(Turbo boost is off)  

ninwavelet version 0.1.0
50 morse wavelet transform
Sampling freq: 1000
1~1000Hz

| Length | back ground | CWT time |
|--------|-------------|----------|
| 1sec   | cupy        | 2.32sec  |
| 1sec   | numpy       | 2.29sec  |
| 50sec  | cupy        | 3.2sec   |
| 50sec  | numpy       | 134sec   |

I do not write about other packages.  
But when I tested, ninwavelets seemd to be extremely faster.  

Did you think ninwavelet based on cuda seems to be extremely fast?  
**It is not true every time.** Throwing data into GPU takes much time.  
But when I compare it to other packages, it still seems to be faster  
even if numpy backend was used.  
And so, I think, ninwavelet is totally, very very fast.  
Ninwavelets is much more simple package now, and so,  
perhaps, it has less functions than other packages.  

## Why is it so fast?

You may think, this package performs strange calculation.  
But what I have done is just adjusting bottle necks.  
Coding wavelet transform itself is not difficult.  
But there is some way to write fast code using numpy or cupy.  
Fast code should skip no purpose calculation.  
This package skips calculation as much as possible.  
Furthur more, transfering data into GPU takes much time.  
I just adjusted the bottle necks carefully.  

## Method

Some mathmaticians may say, DFT is not precise way.  
But we have no good method to perform Fourier transform by digital computer.  
Method of convolve may be good way for Wavelet transform.  
But GMW needs Frourier transform.  
Further more, convolving needs long long loooong time.  
And so, DFT is needed. There may be some methods.  

**1**

```
Convolve(wave, wavelet)  # Very good? But slooooooooooow!
```

**2**

```
iFFT(FFT(wave) * FFT(wavelet))  # Fast, and widely used. But not good. 
```

**3**

``
iFFT(FFT(wave) * FFTed_wavelet)  # Better and faster than 2.
```

I adopted method 3 as Normal mode. Not only GMW, but also Morlet wavelet transform will be performed by 3.

## It is just my hobby
This project is just my hobby, and I am not an engineer or scholar, just a nurd.  
If you cannot believe a nurd without licence of PHD or python engineer, just ignore it.  

# Contribution
I am glad to receive contribution.  
I want to hear constructive oppinions, if I could.  

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
- [x] Logging
- [ ] Bug fix
    + Endless!
