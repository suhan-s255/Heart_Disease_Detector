"""
Setup file for Heart Disease Predictor
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="heart-disease-predictor",
    version="1.0.0",
    author="Sarthaka Mitra",
    description="ML-based heart disease prediction system with Streamlit UI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sarthaka-Mitra/heart-disease-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "pylint>=3.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "heart-disease-predictor=app.streamlit_app:main",
        ],
    },
)
