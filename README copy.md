# BinCP

Code for Robust Conformal Prediction with a Single Binary Certificate Paper

## Overview

This repository contains the code for the paper "Robust Conformal Prediction with a Single Binary Certificate". The code implements the methods described in the paper and provides tools for running experiments and reproducing the results.

## Installation

To install the required dependencies, run the following command:

```bash
pip install statsmodels
pip install -r requirements.txt
```

* In case you want to use the results on sparse smoothing, download the Github repository and install it.

```bash
git clone git+https://github.com/abojchevski/sparse_smoothing.git
cd sparse_smoothing
python setup.py install
```

After installing all the dependecies you can also install the BinCP package.
```bash
cd BinCP
python setup.py install
```

-------------- Up to here is modified.
You can modify the configuration file to change the parameters of the experiments.

## Repository Structure

- `configs/`: Configuration files for the experiments.
- `data/`: Directory to store datasets.
- `scripts/`: Helper scripts for data preprocessing and analysis.
- `src/`: Source code for the implementation.
- `results/`: Directory to store the results of the experiments.

## Citation

If you use this code in your research, please cite the following paper:

```
@inproceedings{your_paper,
  title={Robust Conformal Prediction with a Single Binary Certificate},
  author={Your Name and Co-author Name},
  booktitle={Conference Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [your.email@example.com](mailto:your.email@example.com).
