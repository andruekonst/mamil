import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mamil",
    version="0.0.1",
    author="Andrei V. Konstantinov",
    author_email="andrue.konst@gmail.com",
    description="Multi-Attention Multiple Instance Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andruekonst/mamil",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18.1",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ]
)