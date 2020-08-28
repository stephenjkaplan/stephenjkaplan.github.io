---
layout: post
title: Classifying Accents from Audio Recordings
---

_This blog post details my second project completed while studying at 
[Metis](https://thisismetis.com){:target="_blank"}._

### Project Overview

The guidelines of this project were as follows:
- Create a SQL database to store all tabular data. Make queries from this database to access data while performing 
  analysis and modeling.
- Choose a project objective that requires use of supervised classification algorithms. Experiment with Random Forest, 
  Logistic Regression, XGBoost, K-Nearest Neighbors, and ensembling any combination of models.
- Deploy the model in a [Flask](https://flask.palletsprojects.com/en/1.1.x/){:target="_blank"} application or other 
  interactive visualization. 

It took me a bit longer than previously to decide on a project. I found myself stuck in a mental loop:

1. Search for interesting datasets.
2. Find something promising that would be a suitable binary or multi-classification problem.
3. Realize that the dataset had been downloaded by thousands of people, has been used on large 
   [Kaggle](https://www.kaggle.com/) competitions, etc. 
4. Self-doubt. ("Is my project original enough? Will it stand out enough?")
5. Repeat.

Eventually, I was able to convince myself that as long as I pick something of interest to me and do a good job of 
applying what I had learned, it could be a successful project. What truly pushed me to select a dataset was the time 
limit. This experience taught me the balance between thoughtfully planning a project, but still efficiently delivering a 
useful MVP on a deadline. It's more important to finish a simple yet useful project than come up empty handed with an 
overly complex or over-perfected project.

My project objective was to classify spoken audio of speakers from six countries as American or not-American. (This 
was the result of narrowing the scope of my initial project objective, as explained in [Obstacles](#obstacles) below.)

### The Dataset

mfcc
original paper
mention deploying SQL to AWS EC2

### Obstacles

scoping, lack of data

MFCC interpretability


### Modeling & Results


### Flask App


#### References
Fokoue, E. (2020). [UCI Machine Learning Repository - Speaker Accent Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition). 
Irvine, CA: University of California, School of Information and Computer Science.