from setuptools import setup

setup(name='PyBiplots',
      version='0.1.0',
      description='Package to create some of the most famous biplots methods',
      url='https://github.com/carlostorrescubila',
      author='Carlos A. Torres Cubilla',
      author_email='carlos_t22@usal.es',
      license='MIT',
      packages=['pybiplots'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'sklearn',
          'adjustText'
      ],
      zip_safe=False)