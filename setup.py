from setuptools import setup, find_packages

setup(
    name='ninwavelets',
    version='0.3.0',
    install_requires=['scipy', 'numpy'],
    package_dir={'ninwavelets': 'ninwavelets'},
    packages=find_packages(),
    description='Very fast continuous wavelet transform package',
    long_description='''Brand new Continuous Wavelet Transform package, based on numpy and cupy.
It may be extremely fast.
It can perform various wavelets(Morlet, GMW).''',
    url='https://github.com/uesseu/ninwavelets',
    author='Shoichiro Nakanishi',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
    zip_safe=False,
)
