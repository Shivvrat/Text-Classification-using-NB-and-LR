import SGDClassifier
import bag_of_words
import bernoulli_model
import evaluation_metrics


def evaluate_SGD_bag_of_words(dataset_name):
    """
    This is the method used for evaluation of multinomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: All the evaluation metrics
    """
    # We first import training data for the training
    try:
        spam_email_bag_of_words1, ham_email_bag_of_words1, text_in_all_document, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bag_of_words.convert_to_bag_of_words(
            dataset_name, True)
        spam_email_bag_of_words_test, ham_email_bag_of_words_test, text_in_all_document_test, spam_mail_in_all_documents_test, ham_mail_in_all_documents_test, size_of_total_dataset_test, size_of_spam_dataset_test, size_of_ham_dataset_test, total_file_dictionary_test = bag_of_words.convert_to_bag_of_words(
            dataset_name, False)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    train_data, validation_data = SGDClassifier.divide_into_validation_and_train(spam_email_bag_of_words1,
                                                                                 ham_email_bag_of_words1)
    test_data = SGDClassifier.get_data_from_given_model(spam_email_bag_of_words_test, ham_email_bag_of_words_test)
    words_list = list(train_data[0])
    # we import the train, test and validation datasets
    train_x, train_y = SGDClassifier.convert_data_for_SGD_classifier(train_data, words_list)
    test_x, test_y = SGDClassifier.convert_data_for_SGD_classifier(test_data, words_list)
    valid_x, valid_y = SGDClassifier.convert_data_for_SGD_classifier(validation_data, words_list)
    # In this step we are getting the best parameters for the sklearn SGD classifier
    classifier_model = SGDClassifier.parameter_tuning(valid_x, valid_y)
    # In this step the classifier model is being trained on the training dataset
    trained_classifier_model = SGDClassifier.train_SGD(train_x, train_y, classifier_model)
    # In this step we find the output for the classifier.
    predicted_y, actual_y = SGDClassifier.test_SGD(trained_classifier_model, test_x, test_y)
    # Now calculate the evaluation metrics
    accuracy = evaluation_metrics.accuracy(actual_y, predicted_y)
    precision = evaluation_metrics.precision(actual_y, predicted_y)
    recall = evaluation_metrics.recall(actual_y, predicted_y)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score


def evaluate_SGD_bernoulli_model(dataset_name):
    """
    This is the method used for evaluation of multinomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: All the evaluation metrics
    """
    # We first import training data for the training
    try:
        spam_email_bernoulli_model1, ham_email_bernoulli_model1, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bernoulli_model.convert_to_bernoulli_model(
            dataset_name, True)
        spam_email_bernoulli_model_test, ham_email_bernoulli_model_test, spam_mail_in_all_documents_test, ham_mail_in_all_documents_test, size_of_total_dataset_test, size_of_spam_dataset_test, size_of_ham_dataset_test, total_file_dictionary_test = bernoulli_model.convert_to_bernoulli_model(
            dataset_name, False)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    train_data, validation_data = SGDClassifier.divide_into_validation_and_train(spam_email_bernoulli_model1,
                                                                                 ham_email_bernoulli_model1)
    test_data = SGDClassifier.get_data_from_given_model(spam_email_bernoulli_model_test, ham_email_bernoulli_model_test)
    words_list = list(train_data[0])
    # we import the train, test and validation datasets
    train_x, train_y = SGDClassifier.convert_data_for_SGD_classifier(train_data, words_list)
    test_x, test_y = SGDClassifier.convert_data_for_SGD_classifier(test_data, words_list)
    valid_x, valid_y = SGDClassifier.convert_data_for_SGD_classifier(validation_data, words_list)
    # In this step we are getting the best parameters for the sklearn SGD classifier
    classifier_model = SGDClassifier.parameter_tuning(valid_x, valid_y)
    # In this step the classifier model is being trained on the training dataset
    trained_classifier_model = SGDClassifier.train_SGD(train_x, train_y, classifier_model)
    # In this step we find the output for the classifier.
    predicted_y, actual_y = SGDClassifier.test_SGD(trained_classifier_model, test_x, test_y)
    # Now calculate the evaluation metrics
    accuracy = evaluation_metrics.accuracy(actual_y, predicted_y)
    precision = evaluation_metrics.precision(actual_y, predicted_y)
    recall = evaluation_metrics.recall(actual_y, predicted_y)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score

# evaluate_SGD_bag_of_words(dataset_name) # for bow
# evaluate_SGD_bernoulli_model(dataset_name) # for bm