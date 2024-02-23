import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    

version = "0.0.0"    

REPO_NAME = "end-to-end-ml-project"
AUTHOR_NAME = "melihaltin"
SRC_REPO = "end-to-end-ml-project"
AUTHOR_EMAIL = "melihaltindev@gmail.com"


setuptools.setup(
    name=REPO_NAME,
    version=version,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small example package",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_NAME}/{SRC_REPO}",
    packages=setuptools.find_packages(),
    package_dir={'': 'src'},
)