from pathlib import Path
from typing import Union

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path: Union[str, Path]):
    with open(path, "r") as file:
        return file.read().splitlines()


requirements = read_requirements("requirements.txt")
requirements_dev = read_requirements("requirements_dev.txt")


setuptools.setup(
    name="petals_server",
    version="7.1.0",
    author="Saifeddine ALOUI (ParisNeo)",
    author_email="aloui.saifeddine@gmail.com",
    description="A fastapi server for petals decentralized text generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParisNeo/petals_server",
    packages=setuptools.find_packages(),  
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'petals_server = petals_server.service:main',
        ],
    },
    extras_require={"dev": requirements_dev},
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
