# Mathematics For Data Science – Course Projects
This repository contains a collection of projects developed for the Mathematics for Data Science course at the Isfahan University of Technology (IUT).<br>
The projects focus on the mathematical foundations of Data Science and Machine Learning, including **linear algebra, probability theory, optimization, numerical methods, and their applications in data analysis and predictive modeling**.

All projects were completed under the supervision of **Dr. Ramin Javadi**, and emphasize:
  + Rigorous theoretical derivation of algorithms.
  + Implementation of core methods from scratch (without relying on high-level ML libraries)
  + Analytical evaluation of algorithmic behavior and performance
  + Clear connection between mathematical theory and computational practice

### Table of Contents
[Exercise 1: Geometric and Probabilistic Behavior in High Dimensions](https://github.com/pouriaSameti/Mathematics-For-Data-Science?tab=readme-ov-file#exercise-1-geometric-and-probabilistic-behavior-in-high-dimensions)<br>
[Exercise 2: Spectral Methods and Recommendation Systems](https://github.com/pouriaSameti/Mathematics-For-Data-Science?tab=readme-ov-file#exercise2-spectral-methods-and-recommendation-systems)<br>
[Exercise 3: Exercise 3: PageRank on WikiVote Network](https://github.com/pouriaSameti/Mathematics-For-Data-Science/tree/main?tab=readme-ov-file#exercise-3-pagerank-on-wikivote-network)<br>
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

  
## Exercise 2: Spectral Methods and Recommendation Systems
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


## Exercise 3: PageRank on WikiVote Network
### Objective
This project investigates **graph-theoretic and linear algebraic foundations of ranking algorithms**, with a focus on the **PageRank** method applied to a real-world directed network.

Using the WikiVote dataset, we analyze structural properties of a directed graph and implement multiple versions of the PageRank algorithm. The project emphasizes spectral properties of stochastic matrices, stationary distributions of Markov chains, and efficient large-scale computation using sparse matrix representations.

The main objectives include: 
  + Exploratory analysis of directed network structure
  + Understanding degree distributions and power-law behavior
  + Constructing transition probability matrices
  + Computing stationary distributions via power iteration
  + Handling dangling nodes (dead-ends)
  + Implementing scalable sparse PageRank
  + Extending to Personalized PageRank

### Implementation Details
1. Data Preparation & EDA
   + Load the WikiVote directed graph
   + Compute Number of nodes and edges & In-degree and out-degree distributions
   + Detect self-loops
   + Identify dangling nodes (nodes with out-degree = 0)
   + Display: Top-20 nodes by in-degree & Top-20 nodes by out-degree
   + Generate rank–degree plot on log–log scale to examine potential power-law behavior
2. Dense PageRank on WikiVote Subgraphs
   + Extract and preprocess two subgraphs (A and B)
   + Construct Adjacency matrix & Row-stochastic transition matrix
   + Implement basic PageRank (without dead-end correction)
   + Report convergence behavior and ranking results
3. Dense PageRank (with deadend handling)
   + Modify transition matrix to handle dangling nodes
   + Compute stationary distribution using the Power Method
   + Analyze convergence rate and stability
4. Sparse PageRank (CSR)
   + Construct sparse adjacency matrix using Compressed Sparse Row (CSR) format
   + Perform row normalization to obtain sparse transition matrix P
   + Implement sparse PageRank using power iteration
   + Apply the algorithm to the full WikiVote graph
   + Compare runtime and memory efficiency with dense implementation
5. Personalized PageRank and Dead-End Strategies
   + Implement PageRank with q-fix (teleportation-based correction)
   + Implement PageRank with u-fix (uniform redistribution of dangling mass)
   + Implement Personalized PageRank on the full graph
   + Analyze the effect of personalization vector on ranking results
  
    
