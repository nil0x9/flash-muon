from setuptools import setup, find_packages

setup(
    name="flash_muon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # "torch>=2.5.0",
        "triton>=3.2.0"
    ],
)