from ctypes import POINTER, c_double, c_int, c_float, CDLL
from pathlib import Path
from typing import Optional
from subprocess import run
from ctypes.util import find_library
import numpy as np
import numpy as np
from os import environ, cpu_count
from numpy.ctypeslib import as_ctypes

environ['OMP_NUM_THREADS'] = str(cpu_count())


def load_clib(name: str):
    lib_name = find_library(name)
    print(lib_name)
    return CDLL(lib_name)


class CompileC:
    def __init__(self, fname: str):
        self.fname = fname

    def load(self, name: Optional[str] = None):
        if name is None:
            name = self.make_name()
        return CDLL(name)

    def make_name(self):
        return Path(self.fname).stem + '.dll'

    def recompile(self, out: Optional[str] = None, cpp=True, force=False,
                  optimize_level=3, openmp=True):
        out = self.make_name() if out is None else out
        if Path(out).exists() and not force:
            return self
        compiler = 'g++' if cpp else 'gcc'
        command = [compiler, '-shared', self.fname, '-o', out]
        if optimize_level != 0:
            command.append(f'-O{optimize_level}')
        if openmp:
            command.append('-fopenmp')
        run([compiler, '-shared', self.fname, '-o', out, '-O3', '-fopenmp', '-fPIC'])
        return self


factor = CompileC( Path(__file__).parent / 'c_factor.c').recompile(cpp=False).load().factor
factor.restype = c_int
factor.argtypes = (c_int, c_int)
