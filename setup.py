from setuptools import setup, find_packages

setup(
    name="YOLOv8_Converter",
    version="1.1.0",
    license="MIT",
    author="wh0us",
    python_requires=">=3.9",
    description="Small script for converting/training a YOLOv8 model on a dataset labeled in Label Studio",
    url="https://github.com/wh0us/YOLOv8_Converter/",
    install_requires=["PyYAML>=6.0.1"],
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
)