import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AnnoMate",
    version="1.0.0",
    author="Claudia Chu",
    author_email="cchu@broadinstitute.org",
    description="A general tool to create dashboards for manual review",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/getzlab/AnnoMate",
    project_urls={
        "Bug Tracker": "https://github.com/getzlab/AnnoMate/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".", exclude=['zeption.examples', '']),
    python_requires=">=3.8", # last tested version: 3.9
    install_requires = ['cnv-suite',
                        'dash>=2.11.0',
                        'dash-bootstrap-components',
                        'dash-cytoscape',
                        'dash-daq',
                        'fsspec',
                        'gcsfs',
                        'google-auth',
                        'google-api-core',
                        'hound',
                        'ipykernel',
                        'ipython',
                        'jupyterlab>=4.0.2',
                        'matplotlib',
                        'pandas',
                        'pickleshare',
                        'pillow',
                        'pip',
                        'plotly>=5.15.0',
                        'scipy',
                        'setuptools',
                        'frozendict',
                        # fixes jupyter-dash bug when repeat calls to run_server hangs
                        'flask>=3.0.3',
                        'werkzeug>=2.3.3', # security vulnerability with 2.2.2
                       ]
)   

