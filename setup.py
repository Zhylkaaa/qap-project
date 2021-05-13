from setuptools import setup, find_packages

setup(
    name='QAP',
    version='0.1.0',
    install_requires=['numpy', 'tqdm', 'pandas', 'matplotlib'],
    packages=find_packages(),
)
