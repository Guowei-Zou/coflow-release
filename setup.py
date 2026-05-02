from setuptools import find_packages, setup

setup(
    name="coflow",
    description="CoFlow: coordinated few-step flow for offline multi-agent decision making.",
    author="Guowei Zou, Haitao Wang, Beiwen Zhang, Boning Zhang, and Hejun Wu",
    url="https://github.com/Guowei-Zou/coflow-release",
    project_urls={
        "Project Page": "https://guowei-zou.github.io/coflow/",
    },
    packages=find_packages(include=["diffuser", "diffuser.*"]),
)
