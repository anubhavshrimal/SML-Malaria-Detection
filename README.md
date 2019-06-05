# SML-Malaria-Detection

## Project Overview

This project is done as a part of `Statistical Machine Learning` Course.

In this project our aim is to identify whether a cell is malaria infected or not. We show an in breadth & depth analysis of various features like **HOG, LBP, SIFT, SURF, pixel values** with feature reduction techniques **PCA, LDA** along with normalization techniques such as **z-score** and **min-max** over different classifiers such as **Naive Bayes, SVM XGBoost, Bagging, AdaBoost, K-Nearest Neighbors, Random Forests** and compare their performance by tuning different hyperparameters. We evaluate the performance of these classifiers on metrics such as **Accuracy, Precision, Recall, F1 score and ROC**.

## Dataset

<div style="text-align:center"><img src="plots/dataset_vis.png" height='150px'/></div>

The dataset consists of 27,558 cell images; 13,780 images of infected and uninfected cells each and is taken from the official [NIH Website](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).
You may also download it from [kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

## Algorithm Used

<div style="text-align:center"><img src="plots/training_pipeline.png" /></div>

- Different combinations of feature sets were used, some of which are shown in Table 1 & 2 (**Ugly Duckling Theorem**) many other combinations were tried.
- Evaluated with different classifiers, model parameters were varied using **Grid Search** to find the best parameters (**No Free Lunch Theorem**).
- In PCA, number of components were preserved using **Elbow method over variance of PCA projected data** (Fig. 4).

## Evaluation Metrics and Results

Follwing are the results of the project:

                                            Fig 1. Feature Visualization
   <div style="text-align:center"><img src="plots/feature_visualization.jpg" height='450px'/></div>                                         

                                Fig 1. Comparison between Min-Max and Z-score normalization
   <div style="text-align:center"><img src="plots/normalization_comparison.png" height='250px'/></div>
                                            