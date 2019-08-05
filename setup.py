from setuptools import setup
from setuptools import find_packages


with open("README.rst") as f:
    long_description = f.read()


setup(
    name="sparselearn",
    version="0.0.0",
    license="MIT",
    description=(
        "Efficient sparsity learning in Python"
    ),
    long_description=long_description,
    author="Yngve Mardal Moe",
    author_email="yngve.m.moe@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["numpy", "scikit-learn"],
    extras_require={"test": ["pytest"]},
)
