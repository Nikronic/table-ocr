from setuptools import find_packages
from setuptools import setup
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# version.py defines the VERSION and VERSION_SHORT variables
VERSION: dict = {}
with open('ttocr/version.py', 'r') as version_file:
    exec(version_file.read(), VERSION)

setup(name='ttocr', version=VERSION["VERSION"], packages=find_packages(),
      description='TTOCR: OCRed Table of Tours',
      author='Nikan Doosti',
      author_email='nikan.doosti@outlook.com',
      long_description=long_description,
      long_description_content_type='text/markdown',
      include_package_data = True,
      package_data={
        # ref https://stackoverflow.com/a/73649552/18971263
          '': ['*.json'],
        }
      )