# Setup file for lmdp 
# new setup line added
from setuptools import setup, find_packages
setup(name='lmdp', version='0.1', packages=find_packages())




from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases', 'pytest-cov'],
    'dev': ['pandas==1.3.5', 'plotly==5.5.0', 'wandb']
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='lmdp',
      version='0.0.1',
      description="Library for solving large MDPs.",
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)),
                                      'README.md'), encoding='utf-8').read(),
      url='https://github.com/idigitopia/lmdp',
      author='Aayam K. Shrestha',
      author_email='aayamshrestha@gmail.com',
      license=open(path.join(path.abspath(path.dirname(__file__)),
                             'LICENSE'), encoding='utf-8').read(),
      packages=setuptools.find_packages(),
      install_requires=['pycuda',
                        'munch',
                        'numpy',
                        'tqdm',
                        'sklearn',
                        'gym==0.21.0'],
      include_package_data=True,
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.7',
      classifiers=['Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'])

