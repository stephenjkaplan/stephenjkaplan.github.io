---
layout: post
title: Object Detection for Autonomous Snow Grooming Applications
---

_This post is a technical overview of my [Metis](https://thisismetis.com){:target="_blank"} capstone project, completed 
over a 3 week span. I developed a proof of concept for autonomous snow grooming vehicles at ski resorts. I accomplished 
this by training a [Faster R-CNN](https://arxiv.org/abs/1506.01497){:target="_blank"} object detection model in 
[PyTorch](https://pytorch.org/){:target="_blank"} to detect/classify obstacles in dash cam footage from vehicles moving 
through a ski resort._

The code for this project can be found [here](https://github.com/stephenjkaplan/snow-grooming-object-detection){:target="_blank"}.

### Motivation & Objective

This was a completely open-ended project - we were simply told to choose a "passion project" in any subdomain of 
machine learning. I was overwhelmed with the task of choosing from an infinite project space, but quickly 
asked myself a simple question to narrow down my options: _What machine learning realm did I most want to explore that 
hadn't been heavily covered in the Metis curriculum?_ **Computer Vision**. 

-- INSERT PIC OF MNIST ? --

Yet, even computer vision is a vast field of study. I thought back to the Metis curriculum, when we reviewed a relatively simple application of 
[convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target="_blank"} (CNN) to 
image recognition (via the classic 
[MNIST handwritten digits](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/){:target="_blank"}
dataset). However, I wanted to build on top of that knowledge and challenge myself with something more complex. The 
logical next step seemed to be [object detection](https://en.wikipedia.org/wiki/Object_detection#:~:text=Object%20detection%20is%20a%20computer,in%20digital%20images%20and%20videos){:target="_blank"}
- identifying and locating multiple objects within an image. 

-- INSERT OBJECT DETECTION EXAMPLE WITH AV -- 

I knew that object detection was one component of autonomous vehicle technology, but I felt hesitant. There are a lot of
open source autonomous vehicle datasets out there, but I didn't feel like I could extract any more value than the 
thousands (?) of people with PhD's in computer vision that had already done so.

This was the point at which I considered some of my extracurricular passions. I happen to really enjoy skiing. If you've 
ever been to a ski resort, you might have seen a large vehicle that looks like some combination of a snow plow and a 
military tank. These are "snow cats" - a continuous track vehicle designed to move on rugged, snowy terrain. Snow cats 
are sometimes used for transport, but are mostly used as "snow groomers" at ski resorts. Snow grooming vehicles smooth 
and move around snow to make the mountain safer and more enjoyable for skiers and snowboarders. They operate 
nightly, and also sometimes during the day to keep the mountain open when there is heavy snowfall.

-- INSERT SNOW GROOMING PICTURE --

This felt like it could be a fun, novel application of autonomous vehicle technology. Snow grooming is expensive and 
potentially hazardous - both things that can possibly be reduced through automation. Furthermore, it is a well-defined 
problem in the sense that a snow grooming vehicle only has to be able to detect a narrow subset of obstacles as compared 
to a car driving on a busy city street. Due to the fact that autonomous vehicle software requires many different 
components (instance segmentation, trajectory planning, etc.), I defined my project as a _proof of concept_ for 
autonomous snow grooming vehicles using object detection.

The deliverables of this project are two-fold:
1. A neural network trained to detect objects in images that a snow grooming vehicle might see in its field of view.
2. A demo created by applying the model to snow groomer dash cam footage to draw boundary boxes around detected objects.

### Methodology

#### Object Detection

An object detection model is designed to return two sets of outputs for a given image:
1. The detected instances of semantic objects of a variety of classes in the image (ex. "tree", "person", "car").
2. A set of [Cartesian coordinates](https://mathinsight.org/cartesian_coordinates#:~:text=The%20Cartesian%20coordinates%20(also%20called,distances%20from%20the%20coordinate%20axis.){:target="_blank"} 
   describing the boundary boxes of each detected object in units pixels. 

-- INSERT BOUNDARY BOX IMAGE -- 

Therefore, it follows that the training data for such a model must contain "ground truth" boundary boxes with 
annotated class labels on each image.
   
#### Data

It turns out that it is extremely labor intensive to manually draw boundary boxes on thousands of images. In fact, 
it is common for companies tackling computer vision problems to out source the annotation/labeling task to 
["mechanical turks"](https://www.mturk.com/){:target="_blank"} - a.k.a. humans paid per labeling task they successfully complete. New 
[start-ups](https://scale.com/){:target="_blank"} have sprouted up to specifically provide labeling services.

Luckily, the open-source community once again provides, in the form of pre-annotated datasets for object detection. 
I decided to use [Google's Open Images Dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0jg57){:target="_blank"} 
via the [openimages](https://pypi.org/project/openimages/){:target="_blank"} Python library. This dataset 
contains all images as well as corresponding `.xml` files containing the object classes and boundary boxes for each 
image. This required me to write an [xml parser](https://github.com/stephenjkaplan/snow-grooming-object-detection/blob/master/dataset.py){:target="_blank"}
to extract the boundary box information, but was generally an easy dataset to interact with.

While I initially wanted to detect up to ten different classes, I ran into performance issues that I couldn't remedy
in the short term. The trade-off was: create a model that predicts many object classes with low confidence, or one that 
predicts fewer object classes with higher confidence. I chose the latter option which resulted in three object classes 
for the model to detect:

-- INSERT EXAMPLE OF DATASET WEBSITE -- 

- Person
- Tree
- Ski Lift Pole _(Note: The Google Open Images dataset only had images labeled "street light", which look quite similar 
  to the pole structure for a ski lift, so I assumed it would be a sufficient proxy. I turned out to be right: 
  when presented with only images/video from a ski mountain, the model recognized poles as "street lights". I simply 
  renamed the class label in the `.xml` files from the dataset to "pole".)
  
These three classes seemed like the most essential objects for a snow grooming vehicle to be able to detect as it 
moves up and down the mountain.

#### Transfer Learning

The primary machine learning technique used for this project was 
[transfer learning](https://www.datacamp.com/community/tutorials/transfer-learning?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=295208661496&utm_targetid=dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9028778){:target="_blank"}.
In general, transfer learning involves taking a model already trained on a large dataset for some task and repurposing 
it for another task. More specifically, the process is as follows:
 
1. Take a pre-trained neural network architecture designed to perform 
2. "Freeze" most of the layers in the neural net so that their weights can't be updated when training on a new data set.
3. Replace the final classification layer 

In the context of object detection, 

-- INSERT SOME DIAGRAM OF TRANSFER LEARNING -- 

It may be obvious to some, but what are some reasons you'd want to use transfer learning?

#### Choosing the Right Tools

Determining which computing platform and neural network library to use took up 
the majority of my time while working on this project.


#### Training the Model



#### Evaluating the Model

Quantitative vs. Quantitative



---

_Below is the presentation I gave on this work for my capstone project at Metis. It is a high-level overview of 
what was discussed in this blog post._

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/NhGajxs4t1Q" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>