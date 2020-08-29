---
layout: post
title: Classifying Accents from Audio of Spoken Voice
---

_This blog post details my second project completed while studying at 
[Metis](https://thisismetis.com){:target="_blank"}._ The code for this projec can be found 
[here](https://github.com/stephenjkaplan/speech-accent-classifier).

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

I used the [Speaker Accent Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition#) 
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). This dataset includes data  
extracted from over 300 audio recordings of speakers from six different countries. Half of the data contains speakers 
from the United States, and the other half is divided among Spain, France, Germany, Italy, and the United Kingdom. 
(The original paper is referenced in [References](#references).)

Each audio recording in this dataset was pre-processed and transformed into 12 
[Mel-frequency Cepstrum Coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCC). A long list of 
transformations have to be applied to the audio sample, but an oversimplified definition would be the "power" of the 
signal in each perceptible frequency range.

![MFCC](/images/2020-08-25/mfcc.jpeg)

<small>
It looks good, but what does it mean?
</small>

The data was easy to acquire (a simple .csv download). However, in order to add a learn element to the project, I 
created a SQL database on an AWS EC2 instance to store the data and access it remotely.

### Obstacles

#### Modifying Initial Scope

Similarly to my [last project](https://stephenjkaplan.github.io/2020/07/17/predicting-song-popularity/), I had to 
significantly reduce the scope while working through it. Initially I intended to create a multinomial classifier 
for all of the accents present in the dataset, but the data was too limited. Not only was the total size of the 
dataset particularly small, but this was further compounded by the class imbalance between American accents and all 
other accents. Unsurprisingly, I pivoted towards creating a binary classifier to distinguish between American and 
all other accents.

#### Inability to Reproduce Features

I had hoped to increase the sample size of non-American accents for each by transforming audio from the 
[Speech Accent Archive](https://accent.gmu.edu/) into MFCC features. However, I was unable to reproduce the existing 
data with a couple of different Python packages ([librosa](https://librosa.org/doc/latest/index.html) and 
[python-speech-features](https://python-speech-features.readthedocs.io/en/latest/)). Therefore, I wouldn't have been 
able to trust the results of applying these packages to new audio recordings. This further contributed to the need to 
narrow the scope of the project to a binary classifier.

#### Interpretability of Features

Due to the esoteric nature of the field of psychoacoustics, MFCC features are not interpretable to most people 
(including myself). If I told you that "an accent being American is dependent on the 10th MFCC", that would be 
effectively meaningless. This was unfortunate for two reasons:

1. I couldn't intelligently and creatively do feature engineering to improve my model. Using a more brute force 
   feature engineering approach was somewhat helpful, but having some domain specific knowledge is advantageous.
   
2. The classifier itself wasn't interpretable. In other words, I couldn't draw any useful conclusions on the 
   relationship between particular MFCC and particular accents.

### Modeling & Results

#### Exploratory Data Analysis

The data was of high quality and minimal cleaning was necessary, to I moved quickly into exploratory data analysis.

#### Feature Engineering 

#### Model Selection & Tuning

#### Threshold Tuning

#### Final Scoring

### Flask App


### Summary 



#### References
Fokoue, E. (2020). [UCI Machine Learning Repository - Speaker Accent Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition). 
Irvine, CA: University of California, School of Information and Computer Science.