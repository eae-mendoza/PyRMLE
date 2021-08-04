import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work

if __name__ == "__main__":
    setup(
        name="pyrmle",
        version="1.0.0",
        description="Python package for implementing Regularized Maximum Likelihood for Random Coefficient Models",
        long_description=README,
        url="https://github.com/eae-mendoza/PyRMLE",
        author="Emil Alfred Edgar N. Mendoza",
        author_email="emil.edgar.mendoza@gmail.com",
        license="BSD",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9"
        ],
        packages=["pyrmle","pyrmle_hfuncs"],
        include_package_data=True,
        install_requires=["scipy","sklearn","matplotlib"],
    )