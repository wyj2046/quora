from setuptools import setup, find_packages

setup(name='quora',
      version='0.1',
      packages=find_packages(),
      description='quora',
      author='wyj2046',
      author_email='wyj2046@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)
