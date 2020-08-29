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
MFCC Plot: It looks good, but what does it mean?
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

Before beginning any modeling, I trained the original dataset on K-Nearest Neighbors, Random Forest, and Logistic 
Regression and recorded their train/val ROC AUC scores to serve as a set of baseline models.

#### Exploratory Data Analysis & Feature Engineering

The data was of high quality and minimal cleaning was necessary so I moved quickly into exploratory data analysis.
Upon plotting the distributions of each feature (separated by class), I noticed that some features had bimodal 
distributions for only one class.

![Bimodal](/images/2020-08-25/bimodal.png)

<small>
For features like x5, the distribution for instances where the labeled called was "American"  has a distinct second 
mode. 
</small>

I tried adding an additional boolean feature that indicated if a particular feature was in its respective bimodal range 
to try and put more weight on that behavior in the distributions. Unfortunately this didn't improve the ROC AUC scores 
or overfitting of the baseline models.

I then automatically generated all interaction terms between the original features and plotting their feature 
importances.

![Feature Importance](/images/2020-08-25/feature_importance.png)

<small>
A subset of the interaction features' importances.
</small>

I used the information gained from the feature importance graph to add various interaction features, but eventually 
realized that simply removing `x9` (one of the original MFCC features) yielded the best improvement in ROC AUC 
as compared to the baseline model.

#### Model Selection & Tuning

The next step was to tune the hyperparameters on each baseline model, using the final set of selected features, to 
achieve the best ROC AUC scores and least overfitting for each. This was all done using models in 
[scikit-learn](https://scikit-learn.org/stable/). For the  **Logistic Regression** model, tuning involved:

- Running the model with both L2 (Ridge) and L1 (Lasso) regularization.
- Optimizing the inverse regularization strength to strike a balance between maximizing the ROC AUC validation score 
  and reducing overfitting (minimizing the difference between the training and validation ROC AUC scores.)
  
![ROC AUC](/images/2020-08-25/rocauc.png)

<small>
The difference between training and validation ROC AUC scores with respect to inverse regularization strength.
</small>

![ROC AUC Val](/images/2020-08-25/rocauc_val.png)

<small>
The training and validation scores plotted separately.
</small>

This analysis resulted in a Logistic Regression with L1 regularization and `C = 0.1`, and successfully reduced 
overfitting to a negligible amount while retaining a good ROC AUC score of `0.85`.
 
Tuning **K-Nearest Neighbors (KNN)** was optimized using similar metrics, but instead on the "number of neighbors" 
hyperparameter. The optimal KNN used `6` neighbors and resulted in an ROC AUC score of `0.92`, with reduced overfitting.

Finally, for the Random Forest I tweaked the number of estimators, max depth, and maximum number of observations per 
leaf. The best Forest yielded an ROC AUC score of `0.83`.

Among these three models, the KNN performed the  best. However, I wanted to try ensembling different combinations of 
these three models to see if I could outperform the individual KNN. It turned out that the best model was an ensemble 
of the KNN and Logistic Regression, determined by examining a more detailed set of scores including accuracy, precision, 
and recall of the classifier.

_Bonus: I also played around with XGBoost, but had limited time to optimize the hyperparameters and was satisfied 
enough with the performance of my optimized ensemble._

#### Threshold Tuning

Classification models in scikit-learn by default use a threshold of `0.5` to classify predicted probabilities as the 
positive class (for probabilities above the threshold) or the negative class (for probabilities below). However, by 
changing the threshold, it is possible to get better accuracy out of your model. 

![Threshold Tuning](/images/2020-08-25/threshold_tuning.png)

<small>
Performance of model at different thresholds.
</small>

A threshold of `0.46` yielded negligible change in accuracy while providing a slightly better balance between 
precision and recall.  

#### Final Scoring

Scoring the final model on the test set yielded an overall accuracy of `0.89`. I chose accuracy as the main scoring 
metric because in the case of classifying accents, there's no reason to optimize precision or recall (whereas you 
might care more about these metrics in higher risk domains such as medicine.)

![Confusion Matrix](/images/2020-08-25/confusion_matrix.png)

<small>
Test set confusion matrix of final model.
</small>

### Flask App

I deployed the model via a [Flask application on Google App Engine](https://accent-identification.appspot.com/). 
The app allows you to select audio examples of each accent, listen to the sample, view it's MFCC coefficients, and 
play around with them to yield different predicted classes.

![Flask](/images/2020-08-25/flask.png)

<small>
Screenshot of the Flask application.
</small>

### Summary 

I learned a couple of important things from completing this project:

1. It's important to understand the limitations of your dataset before going too far down an impossible path. In my 
   case, it was wise to quickly switch to a binary classification after realizing how small and imbalanced my 
   dataset was.
   
2. Methodically working through model selection and tuning in an organized fashion can lead you to a model you are 
   happy with. It can help you avoid the infinite loop of "oh, wait, let me just tweak this and try that".

#### References
Fokoue, E. (2020). [UCI Machine Learning Repository - Speaker Accent Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition). 
Irvine, CA: University of California, School of Information and Computer Science.