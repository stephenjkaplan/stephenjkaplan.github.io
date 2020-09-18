---
layout: post
title: Recommender Systems are a Joke: Unsupervised Learning with Stand-Up Comedy
---

_This post documents my first foray into unsupervised learning, natural language processing, and 
recommender systems. I completed this project over a 2-week span during my time as a student at_
[Metis](https://thisismetis.com){:target="_blank"}. 

The code for this project can be found [here](https://github.com/stephenjkaplan/standup-comedy-recommender){:target="_blank"}.
The Flask app I made for this project can be found [here](https://standup-comedy-recommender.herokuapp.com/){:target="_blank"}.

### Intro

A little over halfway through the Metis data science bootcamp, the curriculum shifted from 
[supervised](https://en.wikipedia.org/wiki/Supervised_learning){:target="_blank"} to 
[unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning){:target="_blank"} learning. At that point, my brain was  
firmly in "predict y given X" mode, and it quite honestly took a few days to wrap my head around what it means 
to use machine learning on unlabeled data, and why that would even be useful. It clicked when I applied a 
rudimentary model of the human brain to both of these approaches: often our brains make inferences based 
on our previous experiences and knowledge (supervised learning). However, sometimes we are forced to find previously 
unseen patterns in the world around us (unsupervised learning) before we can make decisions.

Concurrently, we were introduced to 
[Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing){:target="_blank"} (NLP). In addition 
to learning a lot specifically about NLP, one big takeaway was a clearer understanding of what is meant by
"Artificial Intelligence" in the context of machine learning - the simple interpretation being any instance 
of a computer being able to imitate (or even replicate) a human perception or ability. After this realization, 
I went from being somewhat weary of apply machine learning to text/speech, to extremely motivated to 
work on an NLP problem.

As you might expect, the main requirements of this project were:
1. Use unsupervised learning.
2. Use text as the primary data source.

I almost instantly came up with an idea. Stand-up comedy specials are usually in a 1-hour monologue 
format, and are rich and diverse in topics, opinions, colloquialisms, and regional english dialects.
For that reason, and also the fact that I'm a huge fan of the comedy world, I knew that this would be
a stimulating project.

### Data Wrangling

Luckily, I was able to quickly find a website 
([Scraps from the Loft](https://scrapsfromtheloft.com/stand-up-comedy-scripts/){:target="_blank"}) with several hundred 
comedy transcripts. I used the Python [requests](https://requests.readthedocs.io/en/master/){:target="_blank"}) library 
in a [script](https://github.com/stephenjkaplan/standup-comedy-recommender/blob/master/analysis/data_acquisition.py){:target="_blank"} to acquire all of the raw HTML for each transcript, and 
[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/){:target="_blank"} to parse the main text and 
various metadata (comedian, title of comedy special, year). 

After storing the data in two pandas dataframes (one for the metadata and one for the text corpus), I stored 
it in a remote [Mongo](https://www.mongodb.com/){:target="_blank"} database on an Amazon AWS EC2 instance. Admittedly, 
this wasn't really necessary as my corpus was small enough to store locally , but I wanted to get more comfortable with 
both creating MongoDB collections and querying data for analysis and modeling using 
[pymongo](https://pymongo.readthedocs.io/en/stable/){:target="_blank"}.

### Data Cleaning & The NLP Pipeline

As we had been warned, cleaning and preparing the text corpus proved to be the most critical, time 
consuming part of this project. Topic modeling (and clustering) of text data relies on sufficient elimination 
of extraneous words, but also careful inclusion of words that might be indicators of a topic.

First, I performed some "macro" cleaning, removing entire documents (individual transcripts) that would 
throw off the modeling and/or not be relevant in a recommender app. That involved removing:
- Comedy specials in different languages (there were a handful in Italian and Spanish).
- Comedy specials that are not widely available (or available at all) on streaming platforms.
- Short-form comedy such as [monologues on Saturday Night Live](https://www.youtube.com/watch?v=--IS0XiNdpk){:target="_blank"}, 
  or other late night TV shows.

I then created an "[NLP pipeline](https://github.com/stephenjkaplan/standup-comedy-recommender/blob/master/app/nlp_pipeline.py){:target="_blank"}", 
that can take any document from the corpus and perform the following transformations:
- Clean the text by removing punctuation, non-essential parts like the person introducing the comedian, 
  removing numbers, removing line breaks, crazy noises and expressions like "aaaaah", and common English 
  [stop words](https://en.wikipedia.org/wiki/Stop_word){:target="_blank"}.
- "Lemmatized" words to reduce all forms of words to their base or "lemma". (For example, "studies" and 
   "studying" become "study"). The purpose of this is to extract the core meaning and topics out of the text 
   rather than pay attention to how words are being used in the context of a sentence. The Python [NLTK](https://www.nltk.org/){:target="_blank"} library 
   was very useful for this step and several of the previous steps.
- "Vectorized" the entire corpus. In general, this means converting the long strings of text to tabular format, 
   where each column is a word, each row represents a document in the corpus (also known as a 
   [doc-term matrix](https://en.wikipedia.org/wiki/Document-term_matrix){:target="_blank"}), and each cell 
   contains a numerical representation of the words frequency of use or some other metric.
   
The final data format is "fit" on the training data, and then applied to any incoming validation data. 
   
To elaborate a bit on vectorization: I focused on trying out two [types/implementations of vectorization in 
scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text){:target="_blank"}:
1. [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer){:target="_blank"},
   which simply counts the frequency of each word in each document/transcript.
2. [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer){:target="_blank"},
   which calculates the product of term frequency with its inverse-document frequency (a scale of 0 to 1 where values 
   closer to zero are common in the entire dataset of transcripts). This is a valuable metric because it weights 
   words that are common in a document more heavily if they aren't common in general, which is a useful tool for 
   ultimately extracting distinct topics.
   
In general, creating this pipeline was an iterative process. I tried out different transformations and did 
topic modeling (explained in the next section) to evaluate the effectiveness of different components in the pipeline. 
This helped inform my decision to vectorize the text with **TF-IDF**, as it yielded topics that were much 
more discernible and easy to label by looking at each topic's top words.
 
One other note important note is that using pre-built lists of common English stop words to remove from the 
dataset isn't a complete solution. I had to do quite a bit of combing through the transcripts to identify 
insignificant, yet common words and manually add them to the stop words list. It may come as no surprise that 
curse words are extremely prevalent in comedy, and don't usually add much meaning, so I had to include some 
pretty aggressive words in my code (several of which are fairly offensive and I wouldn't use in conversation). 

Aside from also removing names/other irrelevant proper nouns, I also had to carefully decide what to do about racial slurs.
Like it or not, racial slurs are common in stand-up comedy and can sometimes carry important meaning with regards to a 
comedian's jokes. As a result, I left many of them in the dataset, but was also tasked with the uncomfortable task of 
hard-coding some extremely antiquated language in the NLP pipeline class.

### Modeling & Flask App Features

My main objective for this project was to develop a Flask application that provided more nuanced 
recommendations of comedy specials than what's currently available on mainstream streaming platforms. I scoped 
out two features:

1. A dropdown genre filter using genres that were created by machine learning algorithms (as opposed to 
   human labeling.)
2. A search bar that allows a user to describe the comedy they want to see, and get content-based 
   recommendations.

[insert pic]

You can play around with the app [here](https://standup-comedy-recommender.herokuapp.com/){:target="_blank"}.

#### Automatic Genre Filters with Topic Modeling

Creating machine-learned genres involves first applying the unsupervised learning technique of 
[topic modeling](https://en.wikipedia.org/wiki/Topic_model){:target="_blank"}, transforming the resulting 
data to extract which words are most closely related with each topic, and then performing the manual task 
of giving each topic a reasonable label.

In relatively simple terms, topic modeling involves reducing the dimensions of the document-term matrix (words) to a specified number 
of columns (representing the topics, and a weighting for each topic to each document). I tried a few different dimensionality reduction techniques including 
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis){:target="_blank"} (LSA), 
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation){:target="_blank"}, and 
[Non-Negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization){:target="_blank"} (NMF). 
Ultimately, the 
[scikit-learn implementation of NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html){:target="_blank"}.
yielded the most discernible topics.

[insert pic of topics]

~~ TODO talk a bit about labeling the topics and sensitivity ~~

[show screenshot of feature in action]

#### Search Engine Recommender with Content-Based Filtering

- talk about how I technically accomplished this
- show how it work in Flask app

### Thoughts on Language

- share some ethical opinions
- why it's so interesting


ADD IMAGES AFTER
- general comedy pic
- example of doc term matrix
- show topics
flask app pic