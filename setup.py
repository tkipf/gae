from setuptools import setup
from setuptools import find_packages

setup(name='gae',
      version='0.0.1',
      description='Implementation of (Variational) Graph Auto-Encoders in Tensorflow',
      author='Thomas Kipf',
      author_email='thomas.kipf@gmail.com',
      url='https://tkipf.github.io',
      download_url='...',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'networkx',
                        ],
      extras_require={
          'visualization': ['matplotlib'],
      },
      package_data={'gae': ['README.md']},
      packages=find_packages())
