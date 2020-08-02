from setuptools import setup, find_packages

setup(
    name='ninwavelets',
    version='0.1.0',
    install_requires=['scipy', 'numpy'],
    package_dir={'ninwavelets': 'ninwavelets'},
    packages=find_packages(),
    description='Wavelets package',
    long_description='''Analystic wavelets package, based on cuda.
It can perform various wavelets(Morlet, GMW, Shannon).''',
    url='https://github.com/uesseu/ninwavelets',
    author='Forest Segne',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
)
