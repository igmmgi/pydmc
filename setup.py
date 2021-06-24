from setuptools import setup

VERSION = "0.1.0"
DESCRIPTION = "Diffusion Model of Conflict (DMC)"

setup(
    name="pydmc",
    version=VERSION,
    description=DESCRIPTION,
    author="IGM",
    author_email="ian.mackenzie@uni-tuebingen.de",
    install_requires=[
        "numpy",
        "matplotlib",
        "numba",
        "fastKDE",
        "pandas",
        "scipy",
    ],
    packages=["pydmc"],
    include_package_data=True,
    package_data={'': ['data/*.csv']},
    license="MIT",
)
