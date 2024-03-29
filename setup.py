import setuptools


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ofplang",
    version="0.0.1",
    install_requires = requirements,
    packages=setuptools.find_packages(),
    description="The object flow programming language",
    # author="author",
    # author_email="sample@example.com",
    # python_requires='>=3.7',
    )