#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
README = here.joinpath("README.md").read_text()

with open(here.joinpath("requirements.txt")) as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]

setup(
    name="serve-quik",
    version="0.0.2",
    description="functions to make working with TorchServe quik-er",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/donchesworth/serve-quik",
    author="Don Chesworth",
    author_email="donald.chesworth@gmail.com",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=install_requires,
)
