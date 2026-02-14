from setuptools import find_packages, setup

setup(
    name="pydsg",
    version="0.0.1",
    url="",
    author="",
    author_email="",
    description="Central interface for planning pipeline(s)",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
    ],
)

