import setuptools

# Read the contents of the README file as long description for the package
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define the package version
__version__ = "0.0.0"

# Define repository and author information
REPO_NAME = "Kidney-Disease-Classification-Tensorflow"
AUTHOR_USER_NAME = "ommallick02"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "om.mallick02@gmail.com"

# Setup configuration for the package
setuptools.setup(
    name=SRC_REPO,                                 # Package name
    version=__version__,                           # Version of the package
    author=AUTHOR_USER_NAME,                       # Author's username
    author_email=AUTHOR_EMAIL,                     # Author's email address
    description="A package for kidney disease classification using TensorFlow CNN",  # Short description
    long_description=long_description,             # Long description from README.md
    long_description_content="text/markdown",      # Content type of long description
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # Project URL
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",  # URL for reporting issues
    },
    package_dir={"": "src"},                       # Source directory for packages
    packages=setuptools.find_packages(where="src") # Automatically find packages in 'src' directory
)
