from distutils.core import setup

setup(
    name='fuzzycalc',
    version='0.1.0',
    author='sejros',
    author_email='sairos@bk.ru',
    packages=['FuzzyCalc'],
    scripts=[],
    url='http://github.com/sejros/FuzzyCalcPy',
    license='LICENSE.txt',
    description='',
    long_description=open('README.txt').read(),
    install_requires=["matplotlib"],
)
