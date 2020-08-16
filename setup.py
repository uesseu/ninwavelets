from setuptools import setup, find_packages

setup(
    name='ninwavelets',
    version='0.1.1',
    install_requires=['scipy', 'numpy'],
    package_dir={'ninwavelets': 'ninwavelets'},
    packages=find_packages(),
    description='Very fast wavelet transform package',
    long_description='''Brand new analystic wavelets package, based on numpy and cupy.
It may be extremely fast in some cases.
It can perform various wavelets(Morlet, GMW, Shannon).''',
    url='https://github.com/uesseu/ninwavelets',
    author='Forest Segne',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
)
