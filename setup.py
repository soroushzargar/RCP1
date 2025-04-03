import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="bin_cp",
    version="0.1.0",

    author='Soroush H. Zargarbashi',
    author_email='sayed.haj-zargarbashi@cispa.de',
    url='',

    description="Code for paper 'Robust Conformal Prediction with a Single Binary Certificate",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),
    install_requires=[
        'seaborn',
        'ml-collections==0.1.1',
        'sacred',
        'statsmodels',
        'cvxpy',
        "torchattacks",
        "gmpy2",
        # "git+https://github.com/abojchevski/sparse_smoothing.git",
    ],

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='License :: OSI Approved :: MIT License',
)