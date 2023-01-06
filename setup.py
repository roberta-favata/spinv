import setuptools

setuptools.setup(
    name='spinv',
    version='0.1.0', 
    author='Roberta Favata and Antimo Marrazzo',   
    author_email='favata.roberta@gmail.com',
    description='Python package for single-point calculation of topological invariants',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=['pythtb==1.8.0',
                      'tbmodels==1.4.3',],
)