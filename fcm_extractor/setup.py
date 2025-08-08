from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "FCM Extractor - Extract Fuzzy Cognitive Maps from interview transcripts"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="fcm-extractor",
    version="0.1.0",
    author="Maryam Berijanian",
    author_email="berijani@msu.edu",
    description="Extract Fuzzy Cognitive Maps from interview transcripts using NLP and LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/berijani/fcm-extractor",
    project_urls={
        "Bug Reports": "https://github.com/berijani/fcm-extractor/issues",
        "Source": "https://github.com/berijani/fcm-extractor",
        "Documentation": "https://github.com/berijani/fcm-extractor#readme",
    },
    packages=find_packages(include=["fcm_extractor", "fcm_extractor.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fcm-extract=fcm_extractor.run_extraction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fcm_extractor": [
            "config/*.json",
            "config/*.py",
        ],
    },
    zip_safe=False,
    keywords="fuzzy cognitive maps, nlp, llm, clustering, interview analysis, causal relationships",
    platforms=["any"],
) 