def accuracy(true_y, predicted_y):
    """
    This function is used to calculate the accuracy
    :param true_y: These are the given values of class variable
    :param predicted_y: These are the predicted values of the class variable
    :return: the accuracy
    """
    accuracy_count = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each]:
            accuracy_count = accuracy_count + 1
    return accuracy_count / float(len(true_y))


def precision(true_y, predicted_y):
    """
    This function is used to calculate the precision
    :param true_y: These are the given values of class variable
    :param predicted_y: These are the predicted values of the class variable
    :return: the precision
    """
    true_positives = 0
    false_positives = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each] and predicted_y[each] == 1:
            true_positives += 1
        if true_y[each] != predicted_y[each] and predicted_y[each] == 1:
            false_positives += 1
    return true_positives / float(true_positives + false_positives)


def recall(true_y, predicted_y):
    """
    This function is used to calculate the recall
    :param true_y: These are the given values of class variable
    :param predicted_y: These are the predicted values of the class variable
    :return: the recall
    """
    true_positives = 0
    false_negetives = 0
    for each in range(len(true_y)):
        if true_y[each] == predicted_y[each] and predicted_y[each] == 1:
            true_positives += 1
        if true_y[each] != predicted_y[each] and predicted_y[each] == 0:
            false_negetives += 1
    return true_positives / float(true_positives + false_negetives)


def f1_score(recall, precision):
    """
    This function is used to calculate the f1_score
    :param recall: This is the value of recall
    :param precision: This is the value of precision
    :return: the f1_score
    """
    return (2 * recall * precision) / float(recall + precision)
