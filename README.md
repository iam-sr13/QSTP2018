# QSTP2018
## Image Recognition with Deep Learning on Fashion MNIST Dataset
TL;DR: This project is about implementing a Deep Convolutional Neural Network for recognizing Fashion MNIST dataset in PyTorch.
**MAX Accuracy: 94.61 %**

## What is Fashion MNIST?
`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. `Fashion-MNIST` is intended to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Here's an example how the data looks (*each class takes three-rows*):
![](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

Note: For more information on FashionMNIST, kindly visit "https://github.com/zalandoresearch/fashion-mnist"!

## Model Architecture:
The proposed model consist of 5 Convolutional Neural Network layers and 2 Fully Connected (Dense) layers for classification into 10 categories.
Indepth design is described as follows:
![](https://raw.githubusercontent.com/iam-sr13/QSTP2018/master/Accessories/CNNArch.JPG)
                                          ` **Fig 1.0: Proposed Model Architecture** `
                                          
0. **Input Layer**
  * 1x28x28 dimensional Input Tensor
1. **CNN Layer 1**
  * 32 Feature Maps with Kernel size 3x3
    * 32x28x28 dimensional Output Tensor
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.1
2. **CNN Layer 2**
  * 64 Feature Maps with Kernel size 3x3
    * 64x28x28 dimensional Output Tensor
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.1
  * Max Pooling layer of size 2x2
    * 64x14x14 dimensional Output Tensor
3. **CNN Layer 3**
  * 128 Feature Maps with Kernel size 3x3
    * 128x14x14 dimensional Output Tensor
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.1
4. **CNN Layer 4**
  * 256 Feature Maps with Kernel size 3x3
    * 256x14x14 dimensional Output Tensor
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.1
  * Max Pooling layer of size 2x2
    * 256x7x7 dimensional Output Tensor 
5. **CNN Layer 5**
  * 512 Feature Maps with Kernel size 3x3
    * 512x7x7 dimensional Output Tensor
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.1
6. **Dense/Fully Connected Layer 1**
  * 100 Neurons    
  * BatchNorm followed by LeakyReLU Activation
  * Dropout with probability of 0.5  
7. **Dense/Fully Connected Layer 2 : Classification Layer**
  * 10 Neurons    
  * Softmax Activation
  
> * The dataset was preprocessed with normalization and augmented with random horizontal flipping
> * The parameter initialization is He/Kaiming Normal
> * Trained using Cross Entropy Loss over a Softmax Output
> * Optimized using Adam algorithm

## Results:
The Max Accuracy achieved on test dataset using Adam was 94.61%. 
*(Ofcourse using Early Stopping, as it overfitted with further epochs. See the Jupyter notebook for further clarification.)*

## Details on Model Choice:
### Why Preprocessing?
Normalizing the data not only helped in avoiding exploding gradients problem, but also significantly improved the accuracy of the model.
Random flipping was used to virtually double the size of the training data which helped in extracting more accuracy.

### Why He initialization?
If you have any experience with Deep Learning methods, then you ought to know that initial values of weight parameters have significant impact on the output of model and hence on its accuracy. Not going too deep into the mathematics behind the gradients and stuff, any initialization method should help in *symmetry breaking* i.e. it prevents all neurons from behaving identically. 
This particular method is helpful for neurons with Rectified Activations. For sigmoidal neurons, there is another method called Xavier initialization.

### Why the above proposed architecture?
#### 5 CNN layers
I used 5 CNN layers, because unfortunately my machine could only handle upto 5 CNN layers other wise to the infinity and beyond...! :P
Jokes apart, as you know more depth means more accuracy in general in deep learning practice, but we have diminishing returns. (Besides other problems!) I tried many combinations of layers, and found 5 CNN layers were more than enough to fit the training dataset. Also I used 5th CNN layer to push the accuracy above 94%. (Though further Hyperparameter tuning might help in less than 5 layers!)

Kernel size was set to 3x3 because I noticed that most of the misclassified images had very little difference in appearance. (i.e some instances of different categories looked very similar!) My hypothesis was that detecting the very fine local features (instead of bigger differences) will help differentiate the misclassified images. My intuition was correct and it helped in correcting atleast top 1% errors. Though further tuning of hyperparameters should definitely help.

#### Why Dropout?
Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.
This helped avoiding overfitting the model significantly.
![](https://raw.githubusercontent.com/iam-sr13/QSTP2018/master/Accessories/dropout.JPG)
                                          ` **Fig 2.0: Dropout Explained** `

#### Why LeakyReLU?
Using sigmoidal neurons like Tanh first, I faced the vanishing gradient problem and learning was slow as expected. Thus changing to rectified neurons helped significantly to accelerate the learning. But ReLU units have a tendency to die oftenly i.e. they zero out whenever the input is negative. So I opted for Leaky ReLU version just to slightly decrease this zeroing out effect.

#### Why Adam?
I chose Adam, because it converged much faster and achieved higher accuracy than any other optimizer. 
Adam derives from “adaptive moments”, it can be seen as a variant on the combination of RMSProp and momentum, the update looks like RMSProp except that a smooth version of the gradient is used instead of the raw stochastic gradient, the full Adam update also includes a bias correction mechanism.
![](https://raw.githubusercontent.com/iam-sr13/QSTP2018/master/Accessories/plotadamasm.JPG)
                                          ` **Fig 3.0: Using Adam** `
                                          
![](https://raw.githubusercontent.com/iam-sr13/QSTP2018/master/Accessories/plotsgd.JPG)
                                          ` **Fig 4.0: Using SGD** `
                                          
As you can see that SGD is much slower than Adam and has more cost.
Overall I believe that any adaptive moments method is better over other optimization methods.

#### Further Plans!
May try to improve the performance further, and definitely try using more complex architectures if someone gifts me some Titans! ;P


