"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup

setup(
    name="pynattas",
    version="v0.0.0",
    description="Pynattas, a powerful open-source Python package"
    + " that provides a comprehensive set of tools for model building and deployment",
    long_description=open("README.md", encoding="cp437").read(),
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    packages=["pynattas", "pynattas.classes", "pynattas.functions", "pynattas.modules", "pynattas.optimizers"],
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/sirbastiano/PyNA-tta-S"},
)