from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="fias",
    version="0.1.0",
    description="PyTorch implementation scaffold for FIAS medical image segmentation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Xiwei Liu",
    packages=find_packages(include=["fias", "fias.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "nibabel>=5.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ]
    },
)
