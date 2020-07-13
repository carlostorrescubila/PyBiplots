from setuptools import setup
#from setuptools import find_packages

setup(name='pybiplots',
      version='0.1.0',
      author='Carlos A. Torres Cubilla',
      author_email='carlos_t22@usal.es',
      description='Package to performance some biplots methods',
      url='https://github.com/carlostorrescubila',   
      license='MIT',
      packages=['classic'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'seaborn',
          'sklearn',
          'adjustText'
      ],
      #packages=find_packages(),
      #package_dir={'pybiplots': 'pybiplots'}, 
      zip_safe=False
     )
