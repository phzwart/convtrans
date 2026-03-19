from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="local-conv-attention",
    version="0.1.0",
    description="Exact 2D local self-attention implemented as a fixed convolutional-style scaffold.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="OpenAI Codex",
    python_requires=">=3.10",
    packages=find_packages(include=["local_conv_attention", "local_conv_attention.*"]),
    include_package_data=True,
    install_requires=["PyYAML>=6.0"],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev": ["pytest>=8.0"],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
