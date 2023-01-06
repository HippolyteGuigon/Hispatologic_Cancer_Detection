from setuptools import setup, find_packages

setup(
    name='Hispatologic_cancer_detection',
    version='0.1.0',
    packages=find_packages(include=['Hispatologic_cancer_detection', 'Hispatologic_cancer_detection.*']),
    description='Python programm for the Kaggle competition\
        Hispatholohic cancer detecton',
    author='Hippolyte Guigon',
    author_email='Hippolyte.guigon@hec.edu',
    url='https://github.com/HippolyteGuigon/Hispatologic_Cancer_Detection'
)