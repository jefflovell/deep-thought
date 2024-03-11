from setuptools import setup, find_packages

setup (
  name = 'Deep Thought',
  version = '0.9',
  packages = find_packages(),
  install_requires = [
    "numpy >= 1.26.4"
    "torch >= 2.2.1"
  ]
)