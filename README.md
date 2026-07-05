# Probability Distributions and Optimization Coursework

This repository contains the notebooks and supporting files for a formative assessment covering three topics:
Expectation Maximization, Bayesian probability, and gradient descent.

The work is organized as a set of notebook-based demonstrations rather than a single Python package. Each notebook is written to be opened and run in VS Code with a Jupyter kernel.

## Contents

- Part 1: Expectation Maximization on the Galton family height data
- Part 2: Bayesian probability on the IMDb review dataset
- Part 3: Manual gradient descent calculations
- Part 4: Gradient descent implementation in Python

There is also a combined notebook at the repository root that brings the full assessment together.

## Repository Structure

```text
Part 1/
  GaltonFamilies.csv
  EM_Fathers_vs_Children.ipynb
Part 2 Bayes/
  Bayesian_Code.ipynb
  Bayesian Analysis.pdf
Part 3_gradient_descent/
  gradient_descent.ipynb
part4_gradient_descent/
  Gradient_Descent.ipynb
Assembled Jupyter Notebook.ipynb
Contribution tracker
Manual Gradient Descent Calculations
IMDB_Dataset.csv

```

## What Each Part Does

### Part 1: Expectation Maximization

The EM notebooks model the Galton height data as a two-component Gaussian mixture model. The implementation includes:

- Gaussian probability density function
- E-step and M-step
- Log-likelihood tracking
- Initialization strategies
- Convergence checks
- Classification of new height values using posterior probabilities

The detailed version is in [Part 1/EM_Fathers_vs_Children.ipynb](Part%201/EM_Fathers_vs_Children.ipynb).

### Part 2: Bayesian Probability

The Bayesian notebook computes posterior probabilities from the IMDb movie review dataset. It counts keyword occurrences in positive and negative reviews and applies Bayes' theorem to interpret the sentiment impact of selected words.

The main notebook is in [Part 2 Bayes/Bayesian_Code.ipynb](Part%202%20Bayes/Bayesian_Code.ipynb).

### Part 3: Manual Gradient Descent

This section documents the manual gradient descent calculations and the corresponding Python implementation. The parameters are updated step by step across four iterations using matrix operations, with each iteration's gradient and MSE tracked explicitly.

- The full assembled assessment, including this section, is in [Assembled Jupyter Notebook.ipynb](Assembled%20Jupyter%20Notebook.ipynb).
- The handwritten manual calculations are in [Manual Gradient Descent Calculations (Handwritten).pdf](Manual%20Gradient%20Descent%20Calculations%20%28Handwritten%29.pdf).
- The group contribution tracker is in [Machine Learning\_Formative3\_Cohort 3\_Group11 - 1.pdf](Machine%20Learning_Formative3_Cohort%203_Group11%20-%201.pdf).

### Part 4: Gradient Descent in Python

This notebook implements the same gradient descent idea directly in Python, including prediction, error calculation, gradient computation, parameter updates, and convergence visualization.

## Datasets

- [GaltonFamilies.csv](Part%201/GaltonFamilies.csv) is used for the EM analysis.
- [IMDB_Dataset.csv](IMDB_Dataset.csv) is used for the Bayesian probability section.

Both datasets are included in the repository, so the notebooks can run locally without additional downloads.

## Requirements

The notebooks rely on a standard Python data science stack. A typical environment should include:

- numpy
- pandas
- matplotlib
- seaborn
- scipy
- ipykernel

For the Bayesian notebook, only the Python standard library is needed for the core probability calculations, but the dataset exploration cells still benefit from the standard scientific stack.

## Setup

If you are starting from a fresh clone, create a virtual environment and install the notebook dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy ipykernel
```

If VS Code does not select the environment automatically, open the notebook and choose the `.venv` kernel from the kernel picker.

## How to Run

1. Open the repository in VS Code.
2. Select the Python environment that contains the notebook dependencies.
3. Open the notebook you want to run.
4. Run the cells from top to bottom.

If you are using a virtual environment, activate it before launching VS Code or select it manually as the notebook kernel.

## Notes

- The notebooks are intended to be read as worked solutions, not as reusable library code.
- Several cells print intermediate values so the calculations are easy to follow.
- The EM notebook includes both the modeling code and interpretation of the fitted clusters, which makes the assumptions and output easier to verify.

## Summary

This repository demonstrates three core ideas in a classroom-friendly way:

- how EM fits a Gaussian mixture model from scratch,
- how Bayes' theorem can be used to reason about sentiment from observed words,
- and how gradient descent updates parameters iteratively.

The notebooks are self-contained and can be run independently as long as the required datasets and Python packages are available.
