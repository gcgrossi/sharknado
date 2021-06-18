# sharknado

<img src="assets/sharknado_Cover.png" width="85%">

#### _Introduction_

In my effort to learn computer vision with OpenCV and Python I found that a classifier would be a good project to play around with Keras functionalities and Deep Learning libraries/ architectures. And what is better than putting together my passion for Sharks and Data Science and have some good moments? So here is my project of **sharknado: Shark Classification with Python, OpenCV and Keras**.

Before describing a little bit what is inside the repository, I would like to clarify the intention of the project. The idea is not to obtain state of the art precision in shark classification, but revising some important deep learning concept: architectures, hyperparameter tuning, data augmentation, regularisation and so on. For this reason the project is always moving, and I constantly add some tools or test new architectures.

#### Difficulty of the Task

I think is worth spending a couple of words regarding the difficulty of the task. Many books regarding machine learning, as well as blogs present the classification task on "Hallo Word" datasets like (MNIST or Fashion MNIST) or on easily separable datasets, for which a high accuracy can be reached even with simple models and which require not a big effort to train. The task with sharks has an higher level of complicancy because:
- sharks have almost the same shape.
- sharks have almost the same color.
- sharks are often along other sharks.
- under certain angles (i.e. looking from below) is practically impossible to distinguish them.

For this reason I guess that high level features are not really sufficent to distinguish between them, and a some point we will hit a limit posed by:
- The availability of shark images 
- The compromise between the complexity of the architecture used and computing power.

Nonetheless we can try to still reach a reasonable accuracy with not so much computing power, with something that can run on Google Colab without timing-out the runtime.

#### Dataset

For the reasons mentioned above I will try to keep it as simple as possible and I will, for the moment, concentrate my efforts on a binary classifier that will distinguish between white shark and hammerhead shark. Why so? Because at least I could teach the neural network to distinguish the tipical shape of the hammer of the hammerhead shark. Imagine if I started with a distinction between a tiger shark and a white shark. I think I would have miserably failed. As a matter of philosophy of data: you should always start with something simple and doable with your means, and only afterwards increase the complexity. This will give you already a feeling of the complexity of the task and the architecture that best fit your dataset.



