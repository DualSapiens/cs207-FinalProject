import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# with open('requirements.txt') as f:
#    requirements = f.read().splitlines()

setuptools.setup(
    name="therapy_planner",
    version="1.0.0",
    author="DualSapiens",
    author_email="",
    description="A tool for dose optimization for Intensity Modulated Radiation Therapy (IMRT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DualSapiens/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "matplotlib", "gradpy"],
    include_package_data=True
)
