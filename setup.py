from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

VERSION = "1.2.3"

extensions = [
    Extension(
        "mvgarch.optimized.correlation",
        ["mvgarch/optimized/correlation.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="mvgarch",
    version=VERSION,
    author="Jack Tobin",
    author_email="tobjack330@gmail.com",
    description="Multivariate GARCH modelling in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jamesjtobin/mvgarch",
    project_urls={
        "Homepage": "https://github.com/jamesjtobin/mvgarch",
        "Bug Tracker": "https://github.com/jamesjtobin/mvgarch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "arch",
        "pmdarima",
        "scipy",
    ],
    setup_requires=[
        "Cython>=3.0",
        "numpy",
    ],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
