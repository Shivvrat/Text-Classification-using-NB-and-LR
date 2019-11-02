import copy
import random
import numpy as np


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
        each_dict["zero_weight"] = 1
    for each_dict in ham_mail_model:
        each_dict["this_is_the_class_of_the_document"] = 0
        each_dict["zero_weight"] = 1
    all_data = spam_mail_model + ham_mail_model
    # We are using this step to shuffle our data so that different data goes into training and testing everything
    random.shuffle(all_data)
    # 70 percent of the data is for traning and 30 percent of the data is for validation
    train_data = all_data[0: int(len(all_data) * .70)]
    validation_data = all_data[int(len(all_data) * .70): -1]
    return train_data, validation_data


def get_output_for_class(weights, inputs):
    """
    This function is used to get the output for the given weights and inputs
    :param weights: These are the given weights
    :param inputs: These are the given inputs
    :return: We return the sum of product of individual values of weights and inputs
    """
    value = weights['zero_weight'] * 1
    for each in inputs:
        if each == 'this_is_the_class_of_the_document' or each == 'zero_weight':
            continue
        else:
            if each in weights and each in inputs:
                value = value + (weights[each] * inputs[each])
    return value


def get_posterior(weights, inputs):
    """
    This function is used to get of the conditional log likelihood
    :param weights: These are the given weights
    :param inputs: These are the given inputs
    :return: We return the sum of product of individual values of weights and inputs
    """
    value = weights['zero_weight'] * 1
    for each in inputs:
        if each == 'this_is_the_class_of_the_document' or each == 'zero_weight':
            continue
        else:
            if each in weights and each in inputs:
                value = value + (weights[each] * inputs[each])
    return 1 / (float(1 + np.exp(-value)))


def mcap_logistic_regression_train(train_data, total_file_dictionary, eta, lambda_parameter, number_of_iterations):
    """
    This function is used to train the log regression and find the optimum weights for the same
    :param number_of_iterations: These are the number of iteration we want to do for the algorithm
    :param train_data: This is the train data
    :param total_file_dictionary: This is the total list of words in the test data
    :param eta: This is the value of eta
    :param lambda_parameter: This is the value of lambda used for regularization
    :return: We return the optimized weights
    """
    # We are taking w_o outside the array
    weights = copy.deepcopy(total_file_dictionary)
    for each in weights:
        weights[each] = 0
    weights['zero_weight'] = 0
    # Now we update all the weights
    for each in range(number_of_iterations):
        for each_instance in train_data:
            posterior = get_posterior(weights, each_instance)
            sum_of_vals = 0
            for each_weight in weights:
                # Here I checked if the weight is not equal to 0 or not
                if each_instance[each_weight] != 0:
                    # This is the case when w_o is used
                    if each_weight == "zero_weight":
                        sum_of_vals = sum_of_vals + eta * (
                                each_instance["this_is_the_class_of_the_document"] - posterior)
                    else:
                        # This is the case when other w's are used
                        sum_of_vals = sum_of_vals + eta * (each_instance[each_weight] * (
                                each_instance["this_is_the_class_of_the_document"] - posterior))
                    weights[each_weight] = weights[each_weight] + sum_of_vals - eta * lambda_parameter * weights[
                        each_weight]
    return weights


def mcap_logistic_regression_test(test_example, weights):
    """
    This function is used to predict the output for the given test_example
    :param test_example: This is the given test example
    :param weights: These are the given weights
    :return: We return the class of the given instance
    """
    value = get_output_for_class(weights, test_example)
    # 0 is ham and 1 is spam
    if value < 0:
        return 0
    else:
        return 1


def mcap_validation(train_data, validation_data, total_file_dictionary):
    """
    This function is for getting the best value for the parameter lambda
    :param train_data: This is the train data
    :param validation_data:  This is the validation data
    :param total_file_dictionary:  This is the list of all words.
    :return:
    """
    # Here I am doing the grid search for the lambda parameter
    eta = 0.01
    max_accuracy = 0
    best_lambda_value = 2
    # We take the range from 1 increasing 2 at a time
    for each_lambda_value in range(1, 8, 2):
        # We train the algo with the train data
        weights = mcap_logistic_regression_train(train_data, total_file_dictionary, eta, each_lambda_value, 50)
        correct_classification = 0
        # We test on the validation data
        for each_document in validation_data:
            output = mcap_logistic_regression_test(each_document, weights)
            if output == each_document["this_is_the_class_of_the_document"]:
                correct_classification = correct_classification + 1
        accuracy = correct_classification / float(len(validation_data))
        # Here we get the best lambda value
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_lambda_value = each_lambda_value
    return best_lambda_value
