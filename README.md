# Mathematics For Data Science – Course Projects
This repository contains a collection of projects developed for the Mathematics for Data Science course at the Isfahan University of Technology (IUT).<br>
The projects focus on the mathematical foundations of Data Science and Machine Learning, including **linear algebra, probability theory, optimization, numerical methods, and their applications in data analysis and predictive modeling**.

All projects were completed under the supervision of **Dr. Ramin Javadi**, and emphasize:
  + Rigorous theoretical derivation of algorithms.
  + Implementation of core methods from scratch (without relying on high-level ML libraries)
  + Analytical evaluation of algorithmic behavior and performance
  + Clear connection between mathematical theory and computational practice

### Table of Contents
[Exercise 1: Geometric and Probabilistic Behavior in High Dimensions]()<br>
[Exercise 2: Spectral Methods and Recommendation Systems]()<br>
[Exercise 3]()<br>
<br>

### Installation

To run this project, install the required dependencies by executing the following commands:

```python
  pip install numpy
```
```python
  pip install pandas
```
```python
  pip install matplotlib
```
```python
  pip install seaborn
```

## Exercise 1: Geometric and Probabilistic Behavior in High Dimensions

### Objective
This project investigates the geometric and statistical properties of high-dimensional spaces through the analysis of the **unit ball and the unit cube**. The primary goal is to understand how geometric intuition changes as dimensionality increases and how these changes relate to phenomena in machine learning and data science.

The study includes:
  + Generating random points inside the unit ball and unit cube in varying dimensions
  + Analyzing angular distributions between random vectors
  + Studying geodesic distances to the equator and to random hyperplanes
  + Investigating the concentration of measure phenomenon
  + Examining near-orthogonality in high-dimensional spaces
  + Exploring the relationship between high dimensionality, distance behavior, and overfitting
  + Connecting the geometry of the unit ball to the d-dimensional Gaussian distribution

This exercise provides both theoretical insight and empirical validation of core high-dimensional geometry concepts relevant to machine learning.

### Implementation Details
1. Unit Ball Analysis
   + Generate uniformly distributed random points inside the d-dimensional unit ball
   + Compute geodesic distances of points to the equator across different dimensions
   + Analyze geodesic distances to both the equator and randomly generated hyperplanes
   + Study the distribution of angles between random vectors on the unit sphere
   + Compute the mean and variance of angles and distances to the equator
   + Empirically verify the near-orthogonality property in high dimensions
2. Unit Cube Analysis
   + Generate random points inside a d-dimensional unit cube
   + Perform Monte Carlo estimation of the volume of a d-dimensional sphere using sampling within the cube Compare: Volume ratios, Pairwise Euclidean distances and Nearest-neighbor distances
   + Apply random projection for dimensionality reduction and evaluate its effect on pairwise distances
   + Analyze how increasing dimensionality contributes to distance concentration and potential overfitting behavior
3. Johnson–Lindenstrauss (JL) Lemma Experiment
   + This section provides an empirical validation of the Johnson–Lindenstrauss lemma using real-world text data. Dataset: 20 Newsgroups
   + Feature extraction using TF–IDF representation
   + Dimensionality reduction via Gaussian Random Projection
   + Comparison of pairwise Euclidean distances before and after projection
   + Quantitative evaluation of distance preservation under random projection

This experiment demonstrates how high-dimensional data can be embedded into lower-dimensional spaces while approximately preserving geometric structure, which is a fundamental result in high-dimensional probability and machine learning theory.

  
## Exercise2: Spectral Methods and Recommendation Systems
### Objective
This project explores core concepts in linear algebra with a focus on **Singular Value Decomposition (SVD) and Eigenvalue Decomposition (EVD)**, and their application to building a recommendation system.

The primary objective is to design and evaluate a movie recommendation system using matrix factorization techniques. In particular, we apply SVD and Principal Component Analysis (PCA) to a user–item rating matrix and analyze reconstruction quality, dimensionality reduction effects, and predictive performance.

The project emphasizes:
  + Spectral decomposition methods (SVD and EVD)
  + Low-rank matrix approximation
  + Missing data handling strategies
  + Model validation using RMSE
  + Interpretation of latent components

### Implementation Details
1. Data Loading and Structural Overview
   + Construct the user–movie rating matrix A
   + Perform a validation split for performance evaluation
   + Implement two imputation strategies: Zero-fill (missing entries set to 0) & Mean-fill (missing entries replaced with user or global mean)
2. Data Imputation
   + Reconstruct the rating matrix under: Zero imputation & Mean imputation
   + Compare structural differences induced by each approach
3. SVD Reconstruction and Validation RMSE
   + SVD Reconstruction with Fill-zero
   + SVD Reconstruction with Fill-Mean
   + SVD-Mean-Fill Vs SVD-Zero-Fill with respect to RMSE
4. PCA Implementation and Comparison
   + Implement PCA via: Covariance matrix eigendecomposition & Direct SVD-based formulation
   + Apply PCA to: Zero-filled data & Mean-filled data
   + Determine the optimal rank K
   + Compare RMSE across PCA and SVD approaches
   + Analyze differences between covariance-based and direct SVD implementation
5. User Analysis And Recommendation
   + Select the user with the most validation samples
   + Predict ratings for unseen movies (Top-10 recommendation)
   + Interpreting the First Principal Component (PC1)
   + Extract Top-10 positive and negative movies
     
     
