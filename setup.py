from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='ninwavelets',
    version='0.2.0',
    install_requires=['scipy', 'numpy', 'cython'],
    package_dir={'ninwavelets': 'ninwavelets'},
    packages=find_packages(),
    description='Very fast wavelet transform package',
    long_description='''Brand new analystic wavelets package, based on numpy and cupy.
It may be extremely fast.
It can perform various wavelets(Morlet, GMW).''',
    url='https://github.com/uesseu/ninwavelets',
    author='Shoichiro Nakanishi',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
    ext_modules=cythonize("ninwavelets/factor.pyx"),
    zip_safe=False,
)


