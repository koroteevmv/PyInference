# coding=utf-8

from distutils.core import setup
import io
import pyinference


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.txt')


setup(
    name="PyInference",
    version=pyinference.__version__,
    url="https://github.com/sejros/PyInference",
    license='GNU GPL',
    author="sejros",
    author_email="sairos@bk.ru",
    description='Python module for building bayesian and mixed inference nets',
    long_description=long_description,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: Russian',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],

    packages=[
        "pyinference",
        "pyinference.fuzzy",
        "pyinference.inference"
    ],

    install_requires=[
        'numpy>=1.6.1',
        'matplotlib>=1.4.0',
    ]
)
