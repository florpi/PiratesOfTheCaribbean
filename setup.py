from setuptools import find_packages, setup
import distutils.text_file
from pathlib import Path
from typing import List


def parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(
        filename=Path(__file__).with_name(filename)
    ).readlines()


setup(
    name="pirates",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    version="0.1.0",
    description="Mapping disaster risk from aerial imagery",
    author="Eduardo Cuesta-Lazaro and Carolina Cuesta-Lazaro",
    license="MIT",
)
