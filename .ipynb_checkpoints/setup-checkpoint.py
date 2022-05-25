import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JupyterReviewer",
    version="0.0.1",
    author="Claudia Chu",
    author_email="cchu@broadinstitute.org",
    description="A general tool to create dashboards for manual review",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/getzlab/JupyterReviewer",
    project_urls={
        "Bug Tracker": "https://github.com/getzlab/JupyterReviewer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "JupyterReviewer"},
    packages=setuptools.find_packages(where="JupyterReviewer"),
    python_requires=">=3.6",
)   

