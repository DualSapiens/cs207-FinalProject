import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setuptools.setup(
    name="gradpy",
    version="0.0.1",
    author="DualSapiens",
    author_email="",
    description="A package for automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DualSapiens/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"]
)
