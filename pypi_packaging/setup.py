from setuptools import setup, find_packages

# Read in the requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="p3gpt",
    version="0.1.0",
    description="Precious3 GPT - A multimodal language model for biomedical applications",
    author="InSilicoMedicine",
    author_email="info@insilicomedicine.com",
    url="https://github.com/insilicomedicine/precious3-gpt",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
