from setuptools import setup

setup(name='PyBiplots',
      version='0.2.1',
      author='Carlos A. Torres Cubilla',
      author_email='carlos_t22@usal.es',
      description='Package to performance some biplots methods',
      url='https://github.com/carlostorrescubila',   
      license='MIT',
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'seaborn',
          'sklearn',
          'adjustText'
      ],
      packages=['pybiplots'],
      zip_safe=False
     )
