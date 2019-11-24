from setuptools import setup, find_packages

setup(
    name="k-core-embedding",
    version="0.1",
    packages=find_packages('src'),
    package_dir={'': 'src'},

    # metadata to display on PyPI
    authors=["Adrian Jarret", "Pierre Sevestre", "Simon Brandeis"],
)
