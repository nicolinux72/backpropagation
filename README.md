# A Primer on Backpropagation with a Numerical Example,Diagrams, and Python Code

A self-contained introduction to the well-known backpropagation algorithm illustrated step by
step, providing the mathematical elements necessary for understanding and a numerical example
with which to verify what has been learned. A Python script with keras and tensorflow to verify
the calculations performed completes the exposition (available on github).

## Introduction

Backpropagation, short for ’backward propagation of errors,’ is perhaps the most iconic algorithm of
modern machine learning. Suffice it to say that until the mid-1980s, no one believed in the possibility
of training a multilayer network anymore! The story is very interesting and reveals how a mistaken
belief can delay scientific development even by decades.

Although the first simple artificial neural network dates back to Mcculloch and Pitts (1943), it
was in Rosenblatt (1962) that the theory and hardware needed to implement the first neural network
called perceptron was developed. The hardware needed to implement it is depicted in the photos below
where we can see Charles Wightman ( project engineer) adjusting Mark I Perceptron:

Perceptron has several layers but only one trainable so, in modern terms, it is a one-layer neural
network, obviously subject to major limitations formally demonstrated in Minsky and Papert (1969).
And here we come to the exact moment when machine learning based on neural networks was in danger
of being completely abandoned because the two authors firmly believed, although without providing
proof, that the same limitations would also apply to neural networks with more than one layer (deep
learning).

The widespread belief led to a suspension of interest, funding and thus research for the period from
1970 to the first half of the 1980s. Among other things, researchers could not pursue the study of
multilayer models because there was no way to train them: the techniques used by perceptron were
not applicable except for networks with only one layer.

Then, as sometimes happens, a series of innovations contributed to the revival of neural net-
works and, among these, the most relevant was the invention 1 of the backpropagation algorithm by
Rumelhart, Hinton, and Williams (1986). The activation and loss functions had already been made
differential, and backpropagation allows the derivative (more precisely the gradient) of the loss func-
tion to be calculated by moving backwards from the errors (difference between the value predicted
by the model and the actual value) and going up from the last layer to the first. The most relevant
aspect, as we shall see in this article, is the possibility of reusing the calculations already made by the
neural network in its way forward (from the first layer to the last) to calculate the gradient, which
saves resources and makes it possible to handle also even very deep neural networks.

We will begin our exposition by recalling the few elements of mathematics necessary for understand-
ing, which are limited to the multiplication of matrices and vectors and the calculation of derivatives
of linear functions. We will then formally define a multilayer neural network and focus on a numeri-
cal example with two layers. We will use what we have learned to obtain two simple and important
derivatives to be used in the last paragraph to obtain the complete backpropagation algorithm. We
will test our understanding by performing elementary calculations on our example model, and we will
also write a small Python script to verify the correctness of our accounts.

Let us proceed, then :-)
