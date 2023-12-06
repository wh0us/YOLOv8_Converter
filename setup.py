from setuptools import setup, find_packages

setup(
    name="YOLOv8_Converter",
    version="1.0.0",
    license="MIT",
    author="wh0us",
    python_requires=">=3.9",
    description="Small script for converting/training a YOLOv8 model on a dataset labeled in Label Studio",
    url="https://github.com/kotttee/pyrogram_patch/",
    install_requires=["loguru>=0.7.2", "PyYAML>=6.0.1"],
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        'console_scripts': [
            'ydataset=YOLOv8_Converter.converter:main',
        ],
    },
    packages=find_packages(),
)