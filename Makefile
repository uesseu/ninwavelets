python='python3'
server='testpypi'

all:
	$(python) setup.py sdist bdist_wheel
upload:
	$(python) -m twine upload --repository $(server) dist/*
setup:
	$(python) -m pip install --user --upgrade setuptools wheel
	$(python) -m pip install --user --upgrade twine
