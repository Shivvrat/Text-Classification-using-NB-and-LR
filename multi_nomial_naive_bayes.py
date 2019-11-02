from decimal import Decimal
from math import log10 as log


def train_multinomial_NB(spam_email_bag_of_words, ham_email_bag_of_words, text_in_all_document,
                         spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset,
                         size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary):
    """
    This is the main algorithm to train the multinomial Naive Bayes
    :param total_file_dictionary: This is the list containing all the words in the training examples
    :param spam_email_bag_of_words:  This is the list of all words in each spam document (1st value is for first document and so on)
    :param ham_email_bag_of_words: This is the list of all words in each ham document (1st value is for first document and so on)
    :param text_in_all_document: This is the total text in all documents with their frequencies
    :param spam_mail_in_all_documents: This is the total text in all spam documents with their frequencies
    :param ham_mail_in_all_documents: This is the total text in all ham documents with their frequencies
    :param size_of_total_dataset: This is total number of files in all dataset
    :param size_of_spam_dataset: This is total number of files in all spam dataset
    :param size_of_ham_dataset: This is total number of files in all ham dataset
    :return: prior and conditional probability for both spam and ham(all these values are in log)
    """
    no_of_docs = size_of_total_dataset
    # We will first do it for the spam
    no_of_spam_docs = size_of_spam_dataset
    # We create the variables to store the values
    prior = {}
    conditional_probability = {}
    conditional_probability["spam"] = {}
    conditional_probability["ham"] = {}
    conditional_probability_of_non_occurring_word = {}
    conditional_probability_of_non_occurring_word["spam"] = {}
    conditional_probability_of_non_occurring_word["ham"] = {}
    value = Decimal(no_of_spam_docs / float(no_of_docs))
    # First we calculate the priors for the spam and ham dataset
    prior["spam"] = log(value)
    no_of_ham_docs = size_of_ham_dataset
    total_number_of_words_in_ham = sum(ham_mail_in_all_documents.itervalues())
    prior["ham"] = log(no_of_ham_docs / float(no_of_docs))
    total_number_of_words_in_spam = sum(spam_mail_in_all_documents.itervalues())
    # Now we calculate the values for the conditional probabilities
    for each_word in list(spam_mail_in_all_documents):
        conditional_probability["spam"][each_word] = log((spam_mail_in_all_documents[each_word] + 1) / (
            float(total_number_of_words_in_spam + len(text_in_all_document))))

    # Now we will do the same procedure for ham docs
    for each_word in list(ham_mail_in_all_documents):
        conditional_probability["ham"][each_word] = log((ham_mail_in_all_documents[each_word] + 1) / (
            float(total_number_of_words_in_ham + len(text_in_all_document))))
    # These are the values for the conditional probabilities whose words are not in the training dataset
    conditional_probability_of_non_occurring_word["ham"] = log(
        1 / (float(total_number_of_words_in_ham + len(text_in_all_document))))
    conditional_probability_of_non_occurring_word["spam"] = log(
        1 / (float(total_number_of_words_in_spam + len(text_in_all_document))))
    return prior, conditional_probability, conditional_probability_of_non_occurring_word


def test_multinomial_naive_bayes(prior, conditional_probability, conditional_probability_of_non_occurring_word,
                                 an_email_bag_of_words_test):
    """

    :param conditional_probability_of_non_occurring_word: This is the conditional probability for each word in the testing set which is not in the training data
    :param prior: This is the prior for all classes
    :param conditional_probability:  This is the conditional probability for each word in vocabulary in spam and ham data
    :param an_email_bag_of_words_test: This is the given test instance we want to classify
    :return: the class of the given email
    """
    score = {}
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
