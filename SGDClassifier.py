import random

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


def parameter_tuning(validation_x, validation_y):
    """
    This function tunes the parameters for the SGDClassifier and returns the classifier with optimized parameters
    :param validation_x: This is the validation attribute data
    :param validation_y: This is the class values for the given data
    :return: The tuned parameter
    """
    parameters_to_be_tuned = {'alpha': (0.01, 0.05),
                              'max_iter': (range(500, 3000, 1000)),
                              'learning_rate': ('optimal', 'invscaling', 'adaptive'),
                              'eta0': (0.3, 0.7),
                              'tol': (0.001, 0.005)
                              }
    SGDclassifier = SGDClassifier()
    gridSearch = GridSearchCV(SGDclassifier, parameters_to_be_tuned, cv=5)
    gridSearch.fit(validation_x, validation_y)
    return gridSearch


def train_SGD(train_x, train_y, classifier):
    """
    This is the function used to train the Stochastic Gradient Descent Algorithm
    :param train_x: This is train data
    :param train_y: This is the train labels/classes
    :param Classifier: This is the classifier after parameter tuning
    :return: This returns the trained classifier
    """
    return classifier.fit(train_x, train_y)


def test_SGD(trained_classifier, test_x, test_y):
    """
    This function is used to test the given classifier
    :param trained_classifier: This is the trained classifier we have got after training
    :param test_x: This is the test data
    :param test_y: These are the test classes
    :return: We return the accuracy of the given classifier
    """
    predicted_y = []
    for each_document in test_x:
        predicted_y.append(trained_classifier.predict(np.reshape(each_document, (1, -1))))
    return predicted_y, test_y


def convert_data_for_SGD_classifier(data, words_list):
    train_x = []
    train_y = []
    for each_document in data:
        train_x_for_this_document = []
        train_y.append(each_document["this_is_the_class_of_the_document"])
        for each_word in words_list:
            # We are using a try catch here since it may happen that the given word is not in the document
            try:
                train_x_for_this_document.append(each_document[each_word])
            except:
                # If the word is not in the test set then we just 0 as the input for the given word.
                train_x_for_this_document.append(0)
        train_x.append(train_x_for_this_document)
    return train_x, train_y


def get_data_from_given_model(spam_mail_model, ham_mail_model):
    """
    This is the function used to divide the data into test and train data
    :param spam_mail_model: This is the representation(list) of each spam document in the given format
    :param ham_mail_model: This is the representation(list) of each ham document in the given format
    :return: the train and test set
    """
    for each_dict in spam_mail_model:
        each_dict["this_is_the_class_of_the_document"] = 1
    for each_dict in ham_mail_model:
        each_dict["this_is_the_class_of_the_document"] = 0
    all_data = spam_mail_model + ham_mail_model
    # We are using this step to shuffle our data so that different data goes into training and testing everything
    return all_data


def divide_into_validation_and_train(spam_mail_model, ham_mail_model):
    """
    This is the function used to divide the data into test and train data
    :param spam_mail_model: This is the representation(list) of each spam document in the given format
    :param ham_mail_model: This is the representation(list) of each ham document in the given format
    :return: the train and test set
    """
    # Here spam is 1 and ham is 0 (since we are using sigmoid)
    for each_dict in spam_mail_model:
        each_dict["this_is_the_class_of_the_document"] = 1
    for each_dict in ham_mail_model:
        each_dict["this_is_the_class_of_the_document"] = 0
    all_data = spam_mail_model + ham_mail_model
    # We are using this step to shuffle our data so that different data goes into training and testing everything
    random.shuffle(all_data)
    train_data = all_data[0: int(len(all_data) * .70)]
    validation_data = all_data[int(len(all_data) * .70): -1]
    return train_data, validation_data
