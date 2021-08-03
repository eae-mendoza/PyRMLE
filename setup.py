import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="pyrmle",
    version="1.0",
    description="Python package for implementing Regularized Maximum Likelihood for Random Coefficient Models",
    long_description=README,
    url="https://github.com/eae-mendoza/PyRMLE",
    author="Emil Alfred Edgar N. Mendoza",
    author_email="emil.edgar.mendoza@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["scipy","sklearn","matplotlib"],
)