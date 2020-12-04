# NinWavelets
This is a python package for analystic wavelet transform.  
Morlet, Shannon, Generalized Morse(GMW) and so on.  
Generally, CWT is very slow. But this is extremely fast for a python package.  
It is based on Numpy or Cupy, and it may be fast especially on Cupy.  

![My EEG Power!](img/alpha.png)  
This is my alpha band of EEG which was processed by this package.

```python
import cupy as cp
from ninwavelets import Morse
freqs = cp.array(20, 100, 1)

wave = cp.sin(cp.arange(0, 1000, 1))
morse = Morse(1000)
morse.cwt(wave, freqs)
```


# Why NinWavelets?
There may be big advantages and limitations.  
Please see Advantages and Limitations.  

- Use wavelets various wavelets
    + Generalized Morse(A flexible wavelet, which has two parameters.)
    + Morlet/Gabor(Frourier transformed version of Morlet/Gabor.)
    + Shannon(It looks like Haar, when fourier transformed.)
    + Mexcan hat
    + May be more(It is easy to scale!)
- Skipping one FFT when performing CWT.
    + May be better if you use FFT method.
- Speed
    + This package is extremely fast for a pure python package.  
    + Cupy makes it extremely fast.  
        * About One or two order faster than other prevalent packages.
        * Asynchronous calculation by GPU.
    + Even if you can not use cupy, it is very fast.
        * About 3 to 15 times faster than other prevalent packages.
- Reliability
    + Being brand new project, it has no achivement...
    + There may be lots of bugs.
    + Do not rely on it! Read the source code! Test by your self!
- In heavily development
    + Destructive change may be performed.

# Install

From github account.
```bash
pip install git+https://github.com/uesseu/ninwavelets
```

From pypi.
```bash
pip install ninwavelets
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

When you import ninwavelets, it tries to import cupy.  
If cupy could not be imported, it does not raise error but reports it.  

# Usage
- At first, import wavelets and import numpy or cupy.
- The wave must be numpy or cupy data.
  + If you want to use cuda, prepare cupy data! It does not transform automatically!
- Make instance of wavelet.
- Make frequency instance as numpy or cupy.
  + If you did not make it, ninwavelets may becomes slower!
- Perform cwt.


```python
import cupy as cp
from ninwavelets import Morse
freqs = cp.array(20, 100, 1)

wave = cp.sin(cp.arange(0, 1000, 1))
morse = Morse(1000)
morse.cwt(wave, freqs)
# morse.cwt(wave, cp.arange(20, 100, 1))   is little bit slow!

# 2 dimension wave can be converted!
morse.cwt(np.array([wave for n in range(10)], freqs)
```

# Purpose and background
At first, this package was written to perform GMW on mne python.  
But I wrote CWT. Because GMW needs one inverse FFT for no purpose.  
Calculating no purpose is guilt! Now it has own CWT method.  
And I noticed, it may be good for Morlet Wavelet too.  
The result resembles that of mne python.  

This is a brand new project, and under heavily development(Mainly on my Sunday).  
Destructive changes may be made.  

# Exsamples
GMW is similar to morlet wavelet, if you use default param.  
You can calculate complex value, abosolute value and power.  

```python
from ninwavelets import Morse

morse = Morse(1000, gamma=3, beta=17.5)
freq = 60
time = np.arange(0, 0.3, 0.001)

# Now lets analyze sin wave!
sin = np.array([np.sin(time * freq * 2 * np.pi)])

result = morse.power(sin, np.arange(1, 100))
complex_result = morse.cwt(sin, np.arange(1, 100))
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
result = morse.cwt(sin, arange(1, 100))
```


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
InverseFFT(Multiply(FFT(wavelet) @ FFT(wave_to_analyze)))
```

Calculation volume order is $O(NlogN)$

### Fast mode
When Fast mode, it uses fourier transform version of formula.  
Default for Morlet and GMW.  

```
InverseFFT(Multiply(Transformed_wavelet @ FFT(wave_to_analyze)))
```

Calculation volume order is $O(NlogN)$

### Convolve mode
When Convolve, it yields raw wave by inverse fourier transform.  
Default for Haar.

```
Convolve(Wavelet @ wave_to_analyze)
```

Calculation volume order is $O(N^2)$

But, there is some wavelets, which has no formula of raw wave.  
For example, GMW is...  

```
GMW = InverseFFT(raw_formula)
```

In this case, it perform inverse fft before convolving.  

```
Convolve(InverseFFT(Transformed_wavelet) @ wave_to_analyze)
```

Calculation volume order is $O(N^2)$
It seems to be bad.  

### Reverse mode
Even if there is a formula of wavelet, in this mode,  
ninwavelet tries to perform inverseFFT before convolving.  


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

| Param            | Type          | Default |                                                          |
| --               | --            | --      | --                                                       |
| sfreq            | float         | 1000Hz  | Sampling frequency.                                      |
| b                | float         | 17.5    | beta value                                               |
| r                | float         | 3       | gamma value. 3 may be good value.                        |
| real_wave_length | float         | 10      | Length of wavelet(sec). It is modified when CWT.         |
| cuda             | bool          | False   | Use cuda or not. See 'Performance of wavelet transform'. |
| cache_limit      | Optional[int] | 10      | Cache size. If None, cache will be made infinitely.      |

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

## abs
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

## WaveletFormula Class
Super class of formula, which is a member of wavelet classes.  

## WaveletBase Class
Super class of wavelets.  
You can inherit this class and WaveletFormula class, then make new wavelets.  

After inherit WaveletFormula class, you should edit these methods.  

- BaseWavelet.formula
- BaseWavelet.trans_formula
- BaseWavelet.peak_freq
- BaseWavelet.cp_formula
- BaseWavelet.cp_trans_formula

At first, you need to overwrite them.  
They needs to be written by numpy or cupy code.  
Cupy version should start with 'cp'.  

Then, inherit WaveletBase class.  
Make a member variable of WaveletBase named 'formula'.  

## Way to inherit

This is an example.
This code is sub class of BaseWavelet, and is
ninwavelets.MorletWavelet.

```python
import cupy as cp
from ninwavelet import WaveletBase, WaveletFormula
import matplotlib.pyplot as plt

class MorletFormula(WaveletFormula):
    def __init__(self, sigma: float = 7.,
                 gabor: bool = False) -> None:
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


class Morlet(WaveletBase):
    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 real_wave_length: float = 1.,
                 gabor: bool = False, cuda: bool = False,
                 cache_limit: Optional[int] = 10) -> None:
        super(Morlet, self).__init__(sfreq, real_wave_length, cuda,
                                     cache_limit=cache_limit)
        self.formula = MorletFormula(sigma, gabor)
        self.mode = CWTMode.Fast
```

All you should do is write formula and peak_freq.  
The formulas may be written in mathmatical papers! ;)  

# Logging
If you want to use your own logger, you can.  
Methods can catch logging.Logger objects.  

# Advantages and Limitations

## Speed
From version 0.1.3, It became extremely fast **totally**!  
If you want to make best effort, read 'Fast coding of ninwavelets' to write fast code.  

I tested speed of cwt by my NotePC  
'Dell G3 15-3579r' with Intel corei7(4.1Ghz 6core) and Geforce GTX1050.  
(Turbo boost is off)  

ninwavelet version 0.1.0
100 morse wavelet transform
Sampling freq: 1000
30~500Hz

| Length | back ground | CWT time |
|--------|-------------|----------|
| 1sec   | cupy        | 0.205sec |
| 1sec   | numpy       | 1.05sec  |
| 50sec  | cupy        | 1.31sec  |
| 50sec  | numpy       | 134sec   |

I do not write about other packages.  
But when I tested, ninwavelets seemd to be extremely faster totally.  

Did you think ninwavelet based on cuda seems to be extremely fast?  
**It is not true every time.** Throwing data into GPU takes much time.  
In some cases, numpy may faster than cupy.  
For example, if you perform only 10 times, the best way is using numpy.  
Ninwavelets is simple package now, and so,  
it may has less functions than other packages.  

## Fast coding of ninwavelets
### Skipping making wavelets
It is not good for performance to make new wavelets every time,  
if wave length is not changed every time.  
And so, it memorizes wavelets after cwt.  

This is the bad example

```python
morse = Morse(1000)
morse.cwt(wave, np.arange(1, 1000, 1))
```

This is the good example
```python
morse = Morse(1000)
freqs = np.arange(1, 1000, 1)
morse.cwt(wave, freqs)
```

Making wavelets is not so slow and memorizing wavelets may takes big memory.  
And so, optionally, you can skip memorizing.  

When you make an instance, use keyword argument, named 'cache_limit'.  
If cache_limit is 0, it does not remember wavelets.  
If cache_limit is 10, it remebmers 10 wavelets.  
If cache_limit is None, it remembers all the wavelets until it will be deleted.  

Using cache becomes slow, if there is too many cache in the instance.  
Because, looking for cache in the instance takes long time, if there are many wavelets.  
Further more, it uses id, which is built in function of python.  
But in practical usage, it may be almost faster.  
Default value is 10.  

## Why is it so fast?

You may think, this package performs strange calculation.  
But what I have done is just adjusting bottle necks carefully.  
Coding wavelet transform itself is not difficult.  
There is some way to write fast code...

- Skipping calculation as much as possible.
- If possible, skipping tranfering data between main memory and GPU.
- Skipping processing by pure python and write in cupy or numpy as much as possible.
- Use fft.

## There are some anti-patterns
Code is ugly with anti-patterns.  
Unfortunately, these anti-patterns are needed for speed.  
I tested, and good code slowed calculation down.(T_T)  
Such anti-patterns makes it 1.5 times faster.  

Do you like beutiful code? Of course I like it.  
But ninwavelet is an "extreme fast CWT python package" and it is impossible to compromise.  

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

```
iFFT(FFT(wave) * FFTed_wavelet)  # Not good but may be better than 2.
```

I adopted method 3 as Fast mode. Not only GMW, but also Morlet wavelet transform will be performed by 3.

## It is just my hobby
This project is just my hobby, and I am not an engineer or scholar, just a nurd.  
If you cannot believe a nurd without licence of PHD or python engineer,  
just ignore it and enjoy slow life.  

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
    + [x] Decimation
        * It will not be written. It must be a user's work.
        * "Make each program do one thing well."
    + [ ] DWT
        * But, if you want to perform DWT, using np.exp2 may works well.
    + [ ] 2D wavelet
        * [x] FastMode
        * [x] Convolve
- [x] Refactor
    + [x] Do not use fft derived from multiple packages.
- [x] Use cuda or cython and speedup!
    + [x] Cache will be made after making wavelets.
    + [x] Now, It is extremely fast!
- [ ] Kill typos(I am a Jap without intelligence and not good at English) ;(
    + Endless!
- [x] Licence
    + [x] Whether write my name or not.
        * [x] I wrote one of my handle name.
    + [x] Which licence to use.
        * [x] MIT licence.
- [ ] CUPY related limitation
    + Convolve is not allowed in cupy 7.6
    + But, is it a obligation to use convolve?
- [x] Logging
- [ ] Bug fix
    + Endless!
