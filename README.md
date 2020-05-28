# Text Classification using NB and LR
 


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
In this project I have implemented and evaluated Naive Bayes and Logistic Regression for text classification. 

Steps for the project :-
1. Convert the text data into a matrix of features × examples (namely our canonical data representation), using the following approaches. 
    * Bag of Words model: Recall that we use a vocabulary having w words—the set of all unique words in the training set—and represent each email using a vector of word frequencies (the number of times each word in the vocabulary appears in the email).
    * Bernoulli model: As before we use a vocabulary having w words and represent each email (training example) using a 0/1 vector of length w where 0 indicates that the word does not appear in the email and 1 indicates that the word appears in the email.
    
2. Implement the multinomial Naive Bayes algorithm for text classification described here: [http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf](http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf) (see Figure 13.2 in the document). Note that the algorithm uses add-one laplace smoothing. Make sure that you do all the calculations in log-scale to avoid underflow. Use your algorithm to learn from the training set and report accuracy on the test set. Important: Use the datasets generated using the Bag of words model and not the Bernoulli model for this part.
3. Implement the discrete Naive Bayes algorithm we discussed in class. To prevent zeros, use add-one laplace smoothing. Make sure that you do all the calculations in log-scale to avoid underflow. Use your algorithm to learn from the training set and report accuracy on the test set. Important: Use the datasets generated using the Bernoulli model and not the Bag of words model for this part.
4. Implement the MCAP Logistic Regression algorithm with L2 regularization that we discussed in class (see Mitchell’s new book chapter). Try different values of λ. Divide the given training set into two sets using a 70/30 split (namely the first split has 70% of the examples and the second split has the remaining 30%). Learn parameters using the 70% split, treat the 30% data as validation data and use it to select a value for λ. Then, use the chosen value of λ to learn the parameters using the full training set and report accuracy on the test set. Use gradient ascent for learning the weights (you have to set the learning rate appropriately. Otherwise, your algorithm may diverge or take a long time to converge). Do not run gradient ascent until convergence; you should put a suitable hard limit on the number of iterations. Important: Use the datasets generated using both the Bernoulli model and the Bag of words model for this part.
5. Run the SGDClassifier from scikit-learn on the datasets. Tune the parameters (e.g., loss function, penalty, etc.) of the SGDClassifier using GridSearchCV in scikit-learn. Compare the results you obtain for SGDClassifier with your implementation of Logistic Regression. Important: Use the datasets generated using both the Bernoulli model and the Bag of words model for this part.
### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
```
1 import sys
2 import copy 
3 import glob 
4 import os 
5 import re
6 from collections import Counter 
7 import random
8 import numpy as np
9 from decimal import Decimal 
10 from math import log10 as log
11 from sklearn.linear_model import SGDClassifier 
12 from sklearn.model_selection import GridSearchCV 
13 import warnings
```
### Installation

1. Clone the repo
```sh
git clone https://github.com/Shivvrat/Text-Classification-using-NB-and-LR.git
```
Use the main.py to run the algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument:-
```sh
python main.py [dataset_name] [algorithm_name] [type_of_model]
```
Please use the following command line parameters for the main.py file :-
* ***Dataset name***
    Provide the name of folder for the dataset (please keep the folder in the same directory as the code only) For example your folder in which the code is present should look like this :- The other folders like the test, train, ham and spam should have the same folder structure as mentioned in the hw2.pdf file provided on the course home page. For example for spam folder in train directory for enron1 datset should be like directory of the code/enron1/train/spam.
* ***Algorithm name*** 
    * -mnnb - for the multi-nomial naive Bayes (without 3rd argument ) 
    * -dnb - for the discrete naive Bayes (without 3rd argument ) 
    * -mcap - for the Logistic Regression (MCAP) (with 3rd argument ) 
    * -sgd - for the SGD classifier (with 3rd argument )
* ***Type of model***
This is used only for the -mcap and the -sgd algorithms(2nd parameter) 
    * -bow - use this parameter for choosing the bag of words model 
    * -bm - use this for the Bernoulli model
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/Text-Classification-using-NB-and-LR.git](https://github.com/Shivvrat/Text-Classification-using-NB-and-LR.git)
