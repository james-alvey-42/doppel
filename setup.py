from setuptools import setup

setup(
    name="doppel",
    version="0.1.0",
    description="A package for using ML to perform model validation",
    author="James Alvey, Thomas Edwards",
    packages=["doppel"],
    install_requires=[
        "swyft==0.4.5",
        "pytorch_lightning==1.9.5",
        "torch==2.1.2",
        "numpy",
    ],
)
