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
        self.cursor = 0

    def load(self, name: Optional[str] = None):
        if name is None:
            name = self.get_dll_name()
        self.dll = CDLL(Path(name).absolute())
        return self.dll

    def get_dll_name(self):
        return Path(self.fname).stem + '.dll'

    def recompile(self, out: Optional[str] = None,
                  cpp = True, optimize: int = 3,
                  openmp: bool = True):
        out = self.get_dll_name() if out is None else out
        compiler = 'g++' if cpp else 'gcc'
        command = [compiler, '-shared', self.fname, '-o', out, '-fPIC']
        if optimize != 0:
            command.append(f'-O{optimize}')
        if openmp != 0:
            command.append(f'-fopenmp')
        run(command)
        return self

    def find_function(self):
        code = Path(self.fname).read_text()
        start_marker = '/*'
        marker = 'EXPORT PYTHON'
        end_marker = '*/'
        self.cursor = doc_start = to_end_cursor(code, self.cursor, start_marker)-1
        self.cursor = doc_marker = to_end_cursor(code, self.cursor, marker)-1
        self.cursor = doc_end = to_end_cursor(code, self.cursor, end_marker)-1
        doc = code[doc_start: doc_end]
        start = doc_end + len(end_marker)
        end = code.find(')', start)
        func_code, args_code = code.strip()[start: end].split('(')
        func = NameType(*(s.strip() for s in func_code.split(' ')))
        args = [NameType(*(a.strip() for a in arg.split(' ') if a != ''))
                for arg
                in args_code.split(',')]
        func_type = primitive[func.type]
        args_type = [*(primitive[a.type] for a in args)]
        f = self.dll.__getattr__(func.name)
        f.restype = func_type
        f.argtypes = args_type
        f.__name__ = func.name
        f.__doc__ = doc



factor = CompileC( Path(__file__).parent / 'c_factor.c').recompile(cpp=False).load().factor
factor.restype = c_int
factor.argtypes = (c_int, c_int)
