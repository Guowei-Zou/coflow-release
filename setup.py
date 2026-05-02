from setuptools import find_packages, setup

setup(
    name="coflow",
    description="CoFlow: coordinated few-step flow for offline multi-agent decision making.",
    packages=find_packages(include=["diffuser", "diffuser.*"]),
)
