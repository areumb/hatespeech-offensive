from setuptools import find_packages, setup

setup(
    name="hs_generalization",
    version="0.1.0",
    description=(
        "Modeling Offensive Language as a Distinct Class for Hate Speech Detection "
        "(master's thesis project, Kim 2025)"
    ),
    author="Areum Kim",
    python_requires=">=3.9",
    packages=find_packages(include=["hs_generalization", "hs_generalization.*"]),
)
