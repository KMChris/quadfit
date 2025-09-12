from setuptools import setup, Extension, find_packages
import numpy as np
import sys

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-std=c99", "-Wall"]

ext = Extension(
    "quadfit",
    sources=["quadrilateral_fitter/quadfitmodule.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name='quadfit',
    version='1.0.0',
    ext_modules=[ext],
    author='Krzysztof Mizgała',
    author_email='krzysztof@mizgala.pl',
    url='https://github.com/KMChris/quadfit',
    description='QuadrilateralFitter is an efficient and easy-to-use Python library for fitting irregular '
                'quadrilaterals from irregular polygons or any noisy data.',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    keywords='quadrilateral, fitter, polygon, shape analysis, geometry',
    platforms='any',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'shapely',
        'numpy',
        'scipy',
    ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
)