### Overlapping Object Segmentation
#### By: Chris Conte, Kaveh Issapour, and Andrew Sohn
#### Deep Learning Final Project

#### Original Paper

The idea for this Netowrk came from Sara Sabour, Nicholas Frosst, and Geoffrey Hinton's Paper, Dynamic Routing Between Capsules. This paper can be found here: https://arxiv.org/abs/1710.09829 . 

#### Training.

To train run "python main.py". Traning took around two hours on our Google Cloud Instance 

#### Reconstruction and Classification Results

##### Single Letter
![ ](https://imgur.com/gwNabrp.png)
##### MULTI Letter
![ ](https://imgur.com/a/5sTFu.png)
##### Single Letter Classification: Error rate on test set:
![ ](https://imgur.com/oDRMYOJ.png)
##### Single Letter: Loss reconstruction on test set:
![ ](https://imgur.com/295zIRs.png)
##### MULTI Letter Classification: Error rate on test set:
![ ](https://imgur.com/evEX7AO.png)
##### MULTI Letter: Loss reconstruction on test set:
![ ](https://imgur.com/sdYzWj3.png)

#### An Introduction

Human vision excels at both focusing on objects within a noisy environment and identifying object shapes regardless of the orientation, size or angle. The current understanding of how we do so effectively is to build images based on a few carefully selected fixation points. Further, current literature takes the approach of progressively characterizing specific features that we know to belong to different items. Current approaches have used convolutional neural networks to great effect in a limited number of cases. They however struggle in identifying objects in three dimensions that are from different angles, or two dimensional objects that have had severe affine transformations, like those found in CAPTCHA images. They also struggle to identify highly overlapping images, as they are reliant on building a specific set of sub-features for every image. 

Part of this struggle originates from a reliance on ‘max pool’ layers. These have been used alongside convolutional layers. Max pool has a fundamental issue however, which is that by simply taking a maximum value, we are losing a lot of information every time the networks representation of something gets more complex. This results in features of transformed objects that have different orientations, thus different small features, but similar in larger, more conceptual ways to be lost in the lower layers. This paper uses an idea that is illustrated in Hinton’s 2017 paper, dynamic routing, in order to avoid this information loss and thus preserve information until larger conceptual connections can be made. Further, the information loss of max pooling is a detriment obscured images. The features that make up an image can be pooled out to the point where the classifier cannot identify the class of an object.

Instead of a pooling layer then, what is preferable is a kind of routing that includes all of the information understood in the layer below. For this, each convolutional layer is replaced with multiple convolutional layers, the block being labelled a ‘capsule’, and instead of a max pol being executed on scalar values, the vector outputs are routed to the next layer via coupling coefficients, that are optimized during gradient descent. This allows the network to do something that, instead of progressively assigning parts, or features, to wholes and zeroing other possibilities, actually does something closer to inverting the rendering process, where every more information is retained.

Our goal in this paper is to be able to demonstrate a Capsule Network’s powerful ability to classify, with a focus on it’s ability to segment various objects in an image, specifically it’s resilience to transformations that commonly cause convolutional neural networks to falter, and it’s ability to identify images that have high levels of overlap. 

We focus on Capsule Networks facilities which afford it the ability to confront ambiguity as well as conflicting signals, which emerge from segmenting images that are unique due to their complexity. Our work thus delves into generalizing Hinton’s findings to the field Deep Learning which involves object ambiguity, and the specific challenge of being able to classify symbols within CAPTCHA-style images due to their unique complexity and challenges their variation poses to model generalizability. However, we go beyond this task and attempt to add an additional layer of ambiguity by overlaying CAPTCHA images, causing the CAPTCHA-style symbols to overlap and also includes variable affine transformations. We then proceed to classify as well as semantically segment the objects within these images. Our demonstration of the semantic segmentation will occur by employing Capsule Networks ability to classify each symbol within the image and also reconstruct the objects in the image in their entirety which will allow us to evaluate the semantic segmentation. Finally, we add a constraint in this task; to perform the aforementioned task with a significantly reduced dataset in order to build on the findings of Dileep George’s findings on Recursive Cortical Networks.
