server='pypi'

all:
	python3 setup.py sdist bdist_wheel

upload:
	python3 -m twine upload --repository $(server) dist/*

setup:
	python3 -m pip install --user --upgrade setuptools wheel
	python3 -m pip install --user --upgrade twine
