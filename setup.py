from setuptools import setup

setup(
    name="dmc",
    version="0.0.2",
    description="Diffusion Model of Conflict (DMC)",
    author="IGM",
    packages=["dmc"],
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "statsmodels",
        "scipy",
    ],
    zip_safe=False,
)
