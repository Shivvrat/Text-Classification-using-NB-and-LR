import bag_of_words
import evaluation_metrics
import multi_nomial_naive_bayes


def evaluate_multinomial_NB(dataset_name):
    """
    This is the method used for evaluation of multinomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: The method returns the accuracy, precision, recall and f1_score for the given dataset
    """
    # We first import training data for the training
    try:
        spam_email_bag_of_words, ham_email_bag_of_words, text_in_all_document, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bag_of_words.convert_to_bag_of_words(
            dataset_name, True)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    prior, conditional_probability, conditional_probability_of_non_occurring_word = multi_nomial_naive_bayes.train_multinomial_NB(
        spam_email_bag_of_words, ham_email_bag_of_words, text_in_all_document, spam_mail_in_all_documents,
        ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset,
        total_file_dictionary)
    # We now import the data for testing
    spam_email_bag_of_words, ham_email_bag_of_words, text_in_all_document, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bag_of_words.convert_to_bag_of_words(
        dataset_name, False)
    # We calculate the evaluation metric
    # Here we first predict for the spam class and then the ham class
    spam_predict = []
    for each_document in spam_email_bag_of_words:
        spam_predict.append(multi_nomial_naive_bayes.test_multinomial_naive_bayes(prior, conditional_probability,
                                                                                  conditional_probability_of_non_occurring_word,
                                                                                  each_document))
    # We  are taking spam as 1
    spam_actual = [1] * len(spam_predict)
    ham_predict = []
    for each_document in ham_email_bag_of_words:
        ham_predict.append(multi_nomial_naive_bayes.test_multinomial_naive_bayes(prior, conditional_probability,
                                                                                 conditional_probability_of_non_occurring_word,
                                                                                 each_document))
    ham_actual = [0] * len(ham_predict)
    total_actual = spam_actual + ham_actual
    total_predict = spam_predict + ham_predict
    # Now we find the evaluation metrics for the method
    accuracy = evaluation_metrics.accuracy(total_actual, total_predict)
    precision = evaluation_metrics.precision(total_actual, total_predict)
    recall = evaluation_metrics.recall(total_actual, total_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score