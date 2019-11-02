import MCAP_logistic_regression
import bag_of_words
import bernoulli_model
import evaluation_metrics


def evaluate_MCAP_bag_of_words(dataset_name):
    """
    This is the method used for evaluation of multinomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: All the evaluation metrics
    """
    # We first import training data for the training
    try:
        spam_email_bag_of_words1, ham_email_bag_of_words1, text_in_all_document, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bag_of_words.convert_to_bag_of_words(
            dataset_name, True)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    # Firstly we will divide our training data into training and validation data
    train_data, validation_data = MCAP_logistic_regression.divide_into_validation_and_train(spam_email_bag_of_words1,
                                                                                            ham_email_bag_of_words1)
    # Now we find the lambda value by using the grid search algorithm
    lambda_parameter = MCAP_logistic_regression.mcap_validation(train_data, validation_data, total_file_dictionary)
    # Here we merge the training data and the validation data again
    train_data = train_data + validation_data
    alpha_value = 0.01
    # In this step the algorithm learns the weights
    weights = MCAP_logistic_regression.mcap_logistic_regression_train(train_data, total_file_dictionary, alpha_value,
                                                                      lambda_parameter, 500)
    # We now import the data for testing
    spam_email_bag_of_words_test, ham_email_bag_of_words_test, text_in_all_document_test, spam_mail_in_all_documents_test, ham_mail_in_all_documents_test, size_of_total_dataset_test, size_of_spam_dataset_test, size_of_ham_dataset_test, total_file_dictionary_test = bag_of_words.convert_to_bag_of_words(
        dataset_name, False)
    spam_predict = []
    # In this step the algorithm predicts the output for a given dataset
    for each_document in spam_email_bag_of_words_test:
        spam_predict.append(MCAP_logistic_regression.mcap_logistic_regression_test(each_document, weights))
    # We  are taking spam as 1
    spam_actual = [1] * len(spam_predict)
    ham_predict = []
    for each_document in ham_email_bag_of_words_test:
        ham_predict.append(MCAP_logistic_regression.mcap_logistic_regression_test(each_document, weights))
    ham_actual = [0] * len(ham_predict)
    total_actual = spam_actual + ham_actual
    total_predict = spam_predict + ham_predict
    # Now we find the evaluation metrics for the method
    accuracy = evaluation_metrics.accuracy(total_actual, total_predict)
    precision = evaluation_metrics.precision(total_actual, total_predict)
    recall = evaluation_metrics.recall(total_actual, total_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score, lambda_parameter


def evaluate_MCAP_bernoulli_model(dataset_name):
    """
    This is the method used for evaluation of multinomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: All the evaluation metrics
    """
    # We first import training data for the training
    try:
        spam_email_bernoulli_model1, ham_email_bernoulli_model1, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bernoulli_model.convert_to_bernoulli_model(
            dataset_name, True)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    # Firstly we will divide our training data into training and validation data
    train_data, validation_data = MCAP_logistic_regression.divide_into_validation_and_train(spam_email_bernoulli_model1,
                                                                                            ham_email_bernoulli_model1)
    # Now we find the lambda value by using the grid search algorithm
    lambda_parameter = MCAP_logistic_regression.mcap_validation(train_data, validation_data, total_file_dictionary)
    alpha_value = 0.01
    # Here we merge the training data and the validation data again
    train_data = train_data + validation_data
    # In this step the algorithm learns the weights
    weights = MCAP_logistic_regression.mcap_logistic_regression_train(train_data, total_file_dictionary, alpha_value,
                                                                      lambda_parameter, 500)
    # We now import the data for testing
    spam_email_bernoulli_model_test, ham_email_bernoulli_model_test, spam_mail_in_all_documents_test, ham_mail_in_all_documents_test, size_of_total_dataset_test, size_of_spam_dataset_test, size_of_ham_dataset_test, total_file_dictionary_test = bernoulli_model.convert_to_bernoulli_model(
        dataset_name, False)
    spam_predict = []
    # In this step the algorithm predicts the output for a given dataset
    for each_document in spam_email_bernoulli_model_test:
        spam_predict.append(MCAP_logistic_regression.mcap_logistic_regression_test(each_document, weights))
    # We  are taking spam as 1
    spam_actual = [1] * len(spam_predict)
    ham_predict = []
    for each_document in ham_email_bernoulli_model_test:
        ham_predict.append(MCAP_logistic_regression.mcap_logistic_regression_test(each_document, weights))
    ham_actual = [0] * len(ham_predict)
    total_actual = spam_actual + ham_actual
    total_predict = spam_predict + ham_predict
    # Now we find the evaluation metrics for the method
    accuracy = evaluation_metrics.accuracy(total_actual, total_predict)
    precision = evaluation_metrics.precision(total_actual, total_predict)
    recall = evaluation_metrics.recall(total_actual, total_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score, lambda_parameter\

# evaluate_MCAP_bag_of_words(dataset_name) #for bag of words
# evaluate_MCAP_bernoulli_model(dataset_name) # for bernoulli_model