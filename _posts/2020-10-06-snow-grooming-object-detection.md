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
_Note: There is a button within the .ipynb file on GitHub to open it in an interactive Google Colab environment._

### Motivation & Objective

This was a completely open-ended project - we were simply told to choose a "passion project" in any subdomain of 
machine learning. I was overwhelmed with the task of choosing from an infinite project space, but quickly 
asked myself a simple question to narrow down my options: _What machine learning realm did I most want to explore that 
hadn't yet been covered in the curriculum?_ **Computer Vision**. 

![MNIST Dataset](https://miro.medium.com/max/800/1*LyRlX__08q40UJohhJG9Ow.png)

<small>MNIST handwritten digits dataset, commonly used for introducing image recognition. (Image: [Towards Data Science](https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903))</small>

Yet, even computer vision is a vast field of study. I thought back to the Metis curriculum, when we reviewed a relatively simple application of 
[convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target="_blank"} (CNN) to 
image recognition (via the classic 
[MNIST handwritten digits](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/){:target="_blank"}
dataset). However, I wanted to build on top of that knowledge and challenge myself with something more complex. The 
logical next step seemed to be [object detection](https://en.wikipedia.org/wiki/Object_detection#:~:text=Object%20detection%20is%20a%20computer,in%20digital%20images%20and%20videos){:target="_blank"}
- identifying and locating multiple objects within an image. 

![Object Detection Example](https://miro.medium.com/max/1400/1*VXZ8CamGG2Z0M0N4t0Fmng.jpeg)

<small>An example of object detection in autonomous vehicle applications. (Image: [Towards Data Science](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606))</small>

I knew that object detection was one component of autonomous vehicle technology, but I felt hesitant. There are a lot of
open source autonomous vehicle datasets out there, but I didn't feel like I could extract any more value than the 
thousands (?) of people with PhD's in computer vision that had already done so.

This was the point at which I considered some of my extracurricular passions. I happen to really enjoy skiing. If you've 
ever been to a ski resort, you might have seen a large vehicle that looks like some combination of a snow plow and a 
military tank. These are "snow cats" - a continuous track vehicle designed to move on rugged, snowy terrain. Snow cats 
are sometimes used for transport, but are mostly used as "snow groomers" at ski resorts. Snow grooming vehicles smooth 
and move around snow to make the mountain safer and more enjoyable for skiers and snowboarders. They operate 
nightly, and also sometimes during the day to keep the mountain open when there is heavy snowfall.

![Snow Groomer](https://www.pistenbully.com/fileadmin/_processed_/9/4/csm_PB600_4.6_Schraeg_acc1e47cf3.jpg)

<small>A snow groomer. (Image: [Pistenbully](https://www.pistenbully.com/deu/en/level-red.html))</small>

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

![Boundary Box](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pyimagesearch.com%2F2016%2F11%2F07%2Fintersection-over-union-iou-for-object-detection%2F&psig=AOvVaw2zPZ7T3K4MLD1hYEBmRJof&ust=1602196765557000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCOjh05rGo-wCFQAAAAAdAAAAABAD)

<small>Boundary box for a stop sign in an image. (Image: [PyImageSearch](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/))</small>

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

![topics](/images/2020-10-06/tree.png)

<small>Annotated images from Google Open Images Dataset.</small>

- Person
- Tree
- Ski Lift Pole _(Note: The Google Open Images dataset only had images labeled "street light", which look quite similar 
  to the pole structure for a ski lift, so I assumed it would be a sufficient proxy. I turned out to be right: 
  when presented with only images/video from a ski mountain, the model recognized poles as "street lights". I simply 
  renamed the class label in the `.xml` files from the dataset to "pole".)
  
These three classes seemed like the most essential objects for a snow grooming vehicle to be able to detect as it 
moves up and down the mountain. I downloaded approximately 5000 images per class, some containing multiple object instances 
(for instance, 3 annotated trees in a single image). 

#### Transfer Learning

The primary machine learning technique used for this project was 
[transfer learning](https://www.datacamp.com/community/tutorials/transfer-learning?utm_source=adwords_ppc&utm_campaignid=1565261270&utm_adgroupid=67750485268&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=295208661496&utm_targetid=dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9028778){:target="_blank"}.
In general, transfer learning involves taking a model already trained on a large dataset for some task and repurposing 
it for another task. More specifically, the process is as follows:
 
1. Take a pre-trained neural network architecture designed to perform a particular task.
2. "Freeze" most of the layers in the neural net so that their weights can't be updated when training on a new dataset.
3. Replace the final (few) classification layer(s) so that it is both trainable and able to predict the number of 
   classes specific to the new task.
4. Train the model on a custom dataset suited to the new task.

![Transfer Learning](https://pennylane.ai/qml/_images/transfer_learning_general.png)

<small>Transfer Learning (Image: [Pennylane](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html))</small>

It may be obvious to some, _but what are some reasons you'd want to use transfer learning_?

- In the context of image classification/object detection, fine-tuning a neural net pre-trained on a vast image 
  set allows your new model to take advantage of some general network "fundamentals", such as layers that 
  pick up on edges, shapes, and components of a particular object in an image. The final model simply adds a 
  layer that picks up on specific features that separate the various images/objects.
- Since only a fraction of the weights of the neural network have to be updated with every training epoch, the 
  model can be trained much more quickly.

#### Choosing the Right Tools

Determining which computing platform and neural network library to use took up the majority of my time while working on 
this project.

##### Computing Platform

Due to the fact that training neural networks on large datasets can be extremely CPU and RAM intensive, it is 
common to use a GPU on a cloud compute platform rather than a laptop. Unfortunately, I initially jumped to 
an overly complex option. I provisioned a server on Google Cloud (GCP), but quickly realized that this was 
too "heavyweight" and costly for my project, and shut down the server.

I ended up choosing to work with the popular 
[Google Colaboratory](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb){:target="_blank"}. 
This platform, allows you to freely run software on GPUs using a coding interface based on [Jupyter Notebook](https://jupyter.org/){:target="_blank"}, and host 
and access data on a Google Drive folder.

##### Neural Network Toolkit

As most people in the data science community are aware, the two most popular neural network libraries written 
in Python are Google's [TensorFlow](https://www.tensorflow.org/){:target="_blank"} and Facebook's 
[PyTorch](https://pytorch.org/){:target="_blank"}. I initially began to work on this project using Keras (a simplified API within 
the TensorFlow library), simply because that's the library we had used in demos while learning about neural 
networks at Metis. 

I wasn't able to quickly achieve a workable prototype of my model using TensorFlow. I still am not 100% sure 
why this happened, since everything I knew about TensorFlow indicated its vast capabilities. It could have 
been that TensorFlow actually does have a steep learning curve, or that the tutorials available in their 
documentation are better suited for other tasks, or perhaps my lack of understanding in general. Due to the 
extremely limited time allotted for this project, I gracefully took the hit to my ego, and began to search 
around blogs and other documentation for a suitable solution.

I eventually stumbled upon the 
[Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html){:target="_blank"}, 
which turned out to be the perfect starting point, and led me to choose PyTorch as the primary toolkit for this project.
(That being said, there was still a decent amount of work necessary to adapt the tutorial such as ingesting a different 
data format, creating a custom `torch.utils.data.Dataset` class, removing the "Mask R-CNN" model for instance 
segmentation, adding custom image transformations for training, and much more.)

##### Auxiliary Computer Vision Tasks

Demo-ing this model on video footage required the auxiliary task of drawing predicted boundary boxes and labels 
on each frame. The choice to perform this task was a bit more obvious - [OpenCV](https://opencv.org/){:target="_blank"}. OpenCV 
is one of the most popular libraries in the computer vision community. In addition to containing libraries 
for image augmentation, it also has its own machine learning modeling libraries, and more. I used the 
Python implementation of OpenCV, [opencv-python](https://pypi.org/project/opencv-python/){:target="_blank"} to carry out the 
boundary box drawing task.

#### Training the Model

I would have liked to spend a lot more time trying out different optimizers and hyperparameters for the model, 
but given the time constraint I settled on a model with the following features:

- The base model was a Faster R-CNN with [ResNet](http://d2l.ai/chapter_convolutional-modern/resnet.html){:target="_blank"} architecture, 
  pretrained on the [COCO Image Dataset](https://cocodataset.org/#home){:target="_blank"} (>200,000 images containing 
  ~1 million object instances from 80 different object classes).
- The neural network was frozen up until the last 3 backbone layers allowing their weights to be fine-tuned on 
  the custom "ski resort" dataset.
- 80% of the data was allocated to training/validation, and 20% was allocated to testing. Within the 
  training/validation set, it was split 80% for training, 20% for validation.
- The model trained for up to `10` epochs, with a stopping condition in place if the validation score started to 
  rise after continuously falling for some number of epochs.
- I used an initial learning rate of `0.0001` with a learning rate scheduler set to decrease the learning rate 
  by a factor of ten every 3 epochs.
- [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent){:target="_blank"} with a 
  momentum of `0.9` was used as the optimization algorithm. I tried using 
  [ADAM](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/){:target="_blank"}, but that 
  yielded lower performance. 
- The loss function being minimized by SGD is the 
  [Smooth L1 Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss){:target="_blank"} or 
  "Huber" loss.

<p align="center"><img alt="Training Validation Plot" src="/images/2020-10-06/trainval.png"></p>

<small>Training and Validation scores for each of the model's training epochs.</small>

In order to assess overfitting I plotted training loss and validation loss for every epoch. By the time the project 
was due, the model was still too complex. If I had more time, I would have:
- Experimented with introducing [dropout layers](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/){:target="_blank"}.
- Experimented with removing some layers in the network to reduce complexity.
- Trained using a larger dataset.
- Leveraged more compute power to do k-fold cross validation to determine the optimal set of hyperparameters.
- Specifically messed around with weight decay (L2 regularization) to try and reduce model complexity.

#### Evaluating the Model

##### Quantitative Evaluation

As [described earlier](#data), object detection training data is composed of images, boundary boxes, and labels.
A given image may contain multiple detectable objects, and its corresponding annotation will indicate the 
class of those objects (ex. "tree") as well as pixel coordinates indicating the bounding box of each object. 
The boundary box label and coordinates provided in the training/testing data is **ground truth**. Once an 
object detection model is trained, it will output a set of **predicted** boundary boxes for any image it sees.

In order to score the performance of the model, we need some kind of metric to compare **predicted** and 
**ground truth** boundary boxes/labels for every image in the test set. Intersection of Union (IOU) partially 
serves this need by calculating the area of overlap divided by the total area that both boundary boxes take up
in space. If the predicted box were to exactly line up with the ground truth box, then `IOU = 1`.

![IOU](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)

<small>Intersection of Union (IoU) Calculation (Image: [PyImageSearch](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/))</small>

However, this isn't the final metric used to score the performance of the classification model. The 
[evaluator function](https://github.com/stephenjkaplan/snow-grooming-object-detection/blob/master/utilities.py){:target="_blank"}
calculates mean precision and recall across all test images at various IOU's and ranges of IOU's. So, for instance, 
it will output the mean average precision (mAP) for all boundary box predictions that achieved between 0.5:0.95 IOU
with their respective ground truth boundary box. This way of calculating accuracy-type metrics allows for some 
flexibility in how the user chooses to evaluate the results. For instance, if all I care about is generally 
detecting that something exists in an image, I might accept a precision score at an IOU of 0.5. However, if 
that objects exact location in space is important, I might look more at metrics evaluated at an IOU closer to 0.95.

My model achieved a mAP of `0.178` and a mAR of `0.319` for an IOU threshold range of `0.5:0.95`. For reference, 
the original object detection model that I applied transfer learning to achieve a mAP of 0.37 on the COCO 
image dataset.

##### Qualitative Evaluation: The Demo

In order to simulate the performance of the model on an actual snow grooming vehicle, I found some dash cam footage
from snow cats and go pro footage from skiers moving through a mountain. I used OpenCV to separate the videos 
into frames, applied the model to each frame and drew predicted boundary boxes, and stitched the frames 
back together into a video. This was the result:

![Model in Action 1](/images/2020-10-06/demo1.gif)

![Model in Action 2](/images/2020-10-06/demo2.gif)

Qualitatively, the model is able to detect trees, people, and poles. This demo revealed one major point 
of future improvement in the model: it is unable to distinguish "groupings" of objects relative to 
individual instances. It would be important for an autonomous snow grooming vehicle to be able to make this 
distinction. Upon further investigation, it turned out that the dataset I used actually indicated whether
a ground truth bounding box applied to a group or single instance. This is something that could definitely 
be incorporated to improve the model in the future.

### API

With just a bit of time to spare before the final presentation, I decided to prototype an API that contains 
an endpoint for making object detection predictions on any image. I didn't spend time deploying it to a server, but 
it can be run on localhost. (See instructions on how to use it [here](https://github.com/stephenjkaplan/snow-grooming-object-detection){:target="_blank"}.)

---

_Below is the presentation I gave on this work for my capstone project at Metis. It is a high-level overview of 
what was discussed in this blog post._

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/NhGajxs4t1Q" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>