import bernoulli_model
import discrete_naive_bayes
import evaluation_metrics


def evaluate_discrete_NB(dataset_name):
    """
    This is the method used for evaluation of multi-nomial NB on a particular dataset
    :param dataset_name: This is the given dataset name
    :return: We return the accuracy, precision, recall and f1_score for the given dataset
    """
    # We first import training data for the training
    try:
        spam_email_bag_of_words, ham_email_bag_of_words, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bernoulli_model.convert_to_bernoulli_model(
            dataset_name, True)
    except:
        print "You have given wrong file name, please check and run again"
        exit(-1)
    prior, conditional_probability, conditional_probability_of_non_occurring_word = discrete_naive_bayes.discrete_naive_bayes_train(
        spam_email_bag_of_words, ham_email_bag_of_words, spam_mail_in_all_documents,
        ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset,
        total_file_dictionary)
    # We now import the data for testing
    spam_email_bag_of_words, ham_email_bag_of_words, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary = bernoulli_model.convert_to_bernoulli_model(
        dataset_name, False)
    # We calculate the evaluation metric
    # Here we first predict for the spam class and then the ham class
    spam_predict = []
    for each_document in spam_email_bag_of_words:
        spam_predict.append(discrete_naive_bayes.discrete_naive_bayes_test(prior, conditional_probability,
                                                                           conditional_probability_of_non_occurring_word,
                                                                           each_document))
    # We  are taking spam as 1
    spam_actual = [1] * len(spam_predict)
    ham_predict = []
    for each_document in ham_email_bag_of_words:
        ham_predict.append(discrete_naive_bayes.discrete_naive_bayes_test(prior, conditional_probability,
                                                                          conditional_probability_of_non_occurring_word,
                                                                          each_document))
    ham_actual = [0] * len(ham_predict)
    total_actual = spam_actual + ham_actual
    total_predict = spam_predict + ham_predict
    # Now we predict the values for all the evaluation metrics
    accuracy = evaluation_metrics.accuracy(total_actual, total_predict)
    precision = evaluation_metrics.precision(total_actual, total_predict)
    recall = evaluation_metrics.recall(total_actual, total_predict)
    f1_score = evaluation_metrics.f1_score(recall, precision)
    return accuracy, precision, recall, f1_score
# evaluate_discrete_NB(dataset_name)