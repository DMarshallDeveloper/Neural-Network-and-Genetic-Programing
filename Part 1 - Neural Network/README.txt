This folder contains my code for part 1 of the assignment, Neural Networks.
To run my code, after extracting the zip file, simply follow these steps:



I have implemented two versions of the network, one with biases and one without.
My code takes an argument from the command line, if it is 0 the non-biased network will run,
if it is 1 the biased network will run.

RUNNING NON-BIASED NETWORK
- Open a new console and copy and paste (ctrl-SHIFT-v) the following commands into it
- cd 'Downloads/COMP307 Assignment 2 Final/Part 1 - Neural Network/'
- python a2Part1.py 0

RUNNING BIASED NETWORK
- Open a new console and copy and paste (ctrl-SHIFT-v) the following commands into it
- cd 'Downloads/COMP307 Assignment 2 Final/Part 1 - Neural Network/'
- python a2Part1.py 1


I have implemented two versions of the network, one with biases and one without.
My code builds two neural networks, one of each type, and outputs the following for the non-biased first

Prints the output of the first feed-forward pass (both the raw values, and the predicted class),
as well as the new weights after the first back-propagation update. 
It then prints out each epoch number, the weights after the epoch, and the 
accuracy on the training set after the epoch. It then prints out the accuracy 
on the test set after the code has run for 100 epochs. Note that I have implemented 
early stopping in this network, so if it converges before epoch 100 it will stop. 
This will never happen however, since the initial weights and biases are fixed, 
and there is no randomness in this network so there is no variation between runs.

The code should run as expected and generate the output which is stored in sampleoutput.txt

Libraries used:
- numpy
- pandas
- sklearn
- math
To install these libraries if they are absent (shouldn't be needed) use the pip install command
