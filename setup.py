import os
from setuptools import setup, find_packages
from src import NLPTextAugmentation as ba
def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name='NLPTextAugmentation',
    version=ba.__version__,
    author=ba.__author__,
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Text Augmentation',
    long_description=read('README.md'),
    install_requires=['pyvi', 
                      'simalign', 
                      'sentencepiece', 
                      'transformers', 
                      'sacremoses', 
                      'nltk', 
                      'numpy==1.22.0', 
                      'pandas',
                      'gensim==4.3.0',
                      'openpyxl'
                      ],
)