import os
from setuptools import setup, find_packages

setup(
    classifiers=
    ['Programming Language :: Python :: 3',
     "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
     "Operating System :: OS Independent"],
    name='pancollection',
    description="PanCollection based on UDL (https://github.com/XiaoXiao-Woo/UDL)",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    author="XiaoXiao-Woo",
    author_email="wxwsx1997@gmail.com",
    url='https://github.com/XiaoXiao-Woo/PanCollection',
    version='1.0.0',
    include_package_data=True,
    packages=find_packages(exclude=["results", "pancollection/results"]),
    package_data={'pancollection': ['models/*']},
    license='GPLv3',
    python_requires='>=3.7',
    entry_points={             # 如果有命令行工具
        'console_scripts': [
            'accelerate_pansharpening=pancollection.python_scripts.accelerate_pansharpening:hydra_run',
        ],},
    install_requires=[
        "psutil",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tensorboard",
        "addict",
        "yapf",
        "imageio",
        "colorlog",
        # "scipy",
        "h5py",
        "regex",
        "packaging",
        "colorlog",
        "pyyaml",
        "udl-vis",
        "hydra-core"
    ],
)