"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import os

# Load README.md content as the long description
def read_long_description():
    """Reads the long description from the README.md file."""
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Long description not available. Please refer to the GitHub repository."

setup(
    name="pynattas",
    version="0.1.0",  
    description=(
        "Pynattas, a powerful open-source Python package "
        "that provides a comprehensive set of tools for model building and deployment"
    ),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sirbastiano/PyNA-tta-S",
    author="Roberto Del Prete and Andrea Mazzeo",
    author_email="robertodelprete88@gmail.com",
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "torch",
        "torchvision",
        "torchmetrics",
        "pytorch-lightning",
        "tqdm",
        "matplotlib",
        "seaborn",
        "rasterio",
        "tifffile",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "gpu": [
            "torch==1.9.0+cu111",  # For GPU support with CUDA 11.1, adjust version as needed
            "torchvision==0.10.0+cu111",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    packages=find_packages(exclude=["tests*"]),  # Automatically find all packages except tests
    python_requires=">=3.8, <4",
    include_package_data=True,  # Includes non-code files specified in MANIFEST.in
    project_urls={
        "Source": "https://github.com/sirbastiano/PyNA-tta-S",
    },
)