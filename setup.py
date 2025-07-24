from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="momentemu",
    version="1.0.0",
    author="Your Name",  # Update with your name
    author_email="your.email@example.com",  # Update with your email
    description="A lightweight, interpretable polynomial emulator for smooth mappings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzhang0123/MomentEmu",
    py_modules=["MomentEmu"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "scikit-learn",
    ],
    keywords="emulator, polynomial, interpolation, machine-learning, scientific-computing",
)
