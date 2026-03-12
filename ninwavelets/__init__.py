# -*- coding: utf-8 -*-
"""Ninja no Sazanami for extreme high speed wavelet transform.
"""

from .portclang import factor
from .base import (WaveletBase, CWTMode, plot_tf, Baseline, np2cp,
                   WaveletFormula, warmup)
from .nonorthogonal import Morse, Morlet, MexicanHat, Shannon
from .orthogonal import DaubechiesWavelet, daubechies_mra
