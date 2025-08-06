"""Setup script for FCM Extractor package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fcm-extractor",
    version="0.1.0",
    author="Maryam Berijanian",
    author_email="berijani@msu.edu",
    description="Extract Fuzzy Cognitive Maps from interview transcripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fcm-extract=run_extraction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fcm_extractor": [
            "config/*.json",
            "lib/**/*",
        ],
    },
) 