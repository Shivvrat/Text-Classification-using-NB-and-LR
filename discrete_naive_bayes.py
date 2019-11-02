from math import log10 as log


def discrete_naive_bayes_train(spam_email_bernoulli_model, ham_email_bernoulli_model,
                               spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset,
                               size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary):
    """
    
    :param total_file_dictionary: This is the total words list in the train dataset
    :param spam_email_bernoulli_model: This is the list of all the spam documents with bernoulli model
    :param ham_email_bernoulli_model: This is the list of all the ham documents with bernoulli model
    :param spam_mail_in_all_documents:This is the bernoulli model of all the spam documents
    :param ham_mail_in_all_documents:This is the bernoulli model of all the spam documents
    :param size_of_total_dataset: This is total number of files in all dataset
    :param size_of_spam_dataset: This is total number of files in all spam dataset
    :param size_of_ham_dataset: This is total number of files in all ham dataset
    :return: estimate of prior and conditional probability
    """
    no_of_docs = size_of_total_dataset
    no_of_spam_docs = size_of_spam_dataset
    no_of_ham_docs = size_of_ham_dataset
    prior = {}
    # We create variables to store the values
    conditional_probability = {}
    conditional_probability["spam"] = {}
    conditional_probability["ham"] = {}
    conditional_probability_of_non_occurring_word = {}
    conditional_probability_of_non_occurring_word["spam"] = {}
    conditional_probability_of_non_occurring_word["ham"] = {}
    # We calculate the prior of both the classes
    prior["spam"] = log(no_of_spam_docs / float(no_of_docs))
    prior["ham"] = log(no_of_ham_docs / float(no_of_docs))
    # We are doing 1-laplace smoothing and thus we add 1 in the numerator and 2 in denominator(since each word can
    # have two values o, 1 )
    for each_word in spam_mail_in_all_documents:
        conditional_probability["spam"][each_word] = log(
            1 + spam_mail_in_all_documents[each_word] / (float(no_of_spam_docs + 2)))

    for each_word in ham_mail_in_all_documents:
        conditional_probability["ham"][each_word] = log(
            1 + ham_mail_in_all_documents[each_word] / float(no_of_ham_docs + 2))
    # These are the probabilities for the word which are not in the training data and appear in the testing data
    conditional_probability_of_non_occurring_word["ham"] = log(1 / (float(no_of_ham_docs + 2)))
    conditional_probability_of_non_occurring_word["spam"] = log(1 / (float(no_of_spam_docs + 2)))
    return prior, conditional_probability, conditional_probability_of_non_occurring_word


def discrete_naive_bayes_test(prior, conditional_probability, conditional_probability_of_non_occurring_word,
                              an_email_bag_of_words_test):
    """
    This is the function used to generate the output for the naive bayes algorithm
    :param prior: This is the prior generated from the naive bayes
    :param conditional_probability: This is the conditional probability generated from the naive bayes
    :param conditional_probability_of_non_occurring_word: This is the conditional probability of non occurring word generated from the naive bayes
    :param an_email_bag_of_words_test: This is the example on which we are going to test the algo
    :return: The class of the given instance
    """
    score = {}
    # In the following loop we find the words in the given documents for each class and find the posterior
    for each_class in list(prior):
        score[each_class] = prior[each_class]
        for each_word in list(an_email_bag_of_words_test):
            if an_email_bag_of_words_test[each_word] != 0:
                try:
                    score[each_class] += conditional_probability[each_class][each_word]
                # This is the case if the word was not in the train data and thus the laplace pruning gives this result
                except KeyError:
                    score[each_class] += conditional_probability_of_non_occurring_word[each_class]
    # Here we are taking spam as 1 and ham as -1
    if score["spam"] > score["ham"]:
        return 1
    else:
        return 0
