# What do you expect? Revisiting Expectation Formation Models

## Description

This project contains the code and data for my senior thesis, which focuses on evaluating and replicating expectation formation models. Specifically, I replicate three models:

- Sticky-information model
- Noisy-information model
- Diagnostic expectations model

## Directories

- **cleaned_data/**: Contains cleaned datasets used in the analysis.
- **code/**: Contains Python scripts for data cleaning, summary statistics, estimation, plotting, and robustness checks.
- **data/**: Contains raw datasets.
- **output/**: Directory for storing output files such as plots and results.

## Key Files

- **code/_01_cleaning.py**: Script for cleaning the raw data.
- **code/_02_sumstats.py**: Script for generating summary statistics.
- **code/_03_estimation.py**: Script for model estimation.
- **code/_04_plots.py**: Script for generating plots.
- **code/_05_robustnesschecks.py**: Script for performing robustness checks.

## Installation

To run the code in this repository, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- statsmodels

You can install the required packages using pip:

```sh
pip install pandas numpy matplotlib statsmodels
```

## Usage

1. **Data Cleaning**: Run the `_01_cleaning.py` script to clean the raw data.
2. **Summary Statistics**: Run the `_02_sumstats.py` script to generate summary statistics.
3. **Model Estimation**: Run the `_03_estimation.py` script to estimate the models.
4. **Plotting**: Run the `_04_plots.py` script to generate plots.
5. **Robustness Checks**: Run the `_05_robustnesschecks.py` script to perform robustness checks.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or inquiries, please contact Calvin McElvain at [mcelvain1@hotmail.com].
