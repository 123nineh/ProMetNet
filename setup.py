from distutils.core import setup

requirements = [
    "numpy",
    "networkx==3.2.1",
    "pandas",
    "torch==2.5.1+cu121",
    "lightning==2.3.1",
    "shap==0.44.1",
    "matplotlib==3.7.2",
    "plotly==5.18.0",
    "nbformat==5.10.4",
    "kaleido==1.0.0",
]

from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="ProMetNet",
    version="0.1.0",
    description="A novel pathway-driven and interpretable integration framework based on functional interactions between protemics and metabolomics",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Minghui Zhao",
    author_email="zhaominghui@mail.sdu.edu.cn.com",

    license="MIT",
    url="https://github.com/XXXX/ProMetNet",

    packages=["ProMetNet"],
    install_requires=requirements,

    python_requires=">=3.10",
)
