import copy
import glob
import os
import re
from collections import Counter


def import_train_data(data_set_name, train_data_type):
    """
    This function takes the dataset name and returns the files for the spam and ham text files for the train data
    :param train_data_type: In this if it is true then we have train data else test data
    :param data_set_name: This is the dataset name
    :return: We return two lists in which consists of the spam and ham for given dataset
    """
    files_ham = []
    files_spam = []
    total_data = ""
    path = os.path.join(os.getcwd(), data_set_name)
    if train_data_type == True:
        path = os.path.join(path, "train")
    else:
        path = os.path.join(path, "test")
    path_ham = os.path.join(path, "ham")
    path_spam = os.path.join(path, "spam")
    files_list_spam = glob.glob(path_spam + "/" + "*.txt")
    files_list_ham = glob.glob(path_ham + "/" + "*.txt")
    # Here we save the files in given folder to a list and then read them
    for spam_files in files_list_spam:
        files_spam.append(open(spam_files, "r").read())
        total_data = total_data + " " + open(spam_files, "r").read()
    for ham_files in files_list_ham:
        files_ham.append(open(ham_files, "r").read())
        total_data = total_data + " " + open(ham_files, "r").read()
    # we find the size of the dataset and the number of instances with spam and number of instances with ham
    size_of_total_dataset = len(files_list_ham) + len(files_list_spam)
    size_of_ham_dataset = len(files_list_ham)
    size_of_spam_dataset = len(files_list_spam)
    return files_spam, files_ham, total_data, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset


def convert_to_bag_of_words(dataset_name, train_data_type):
    """
    This function returns the bag of words for given dataset
    :param dataset_name: This is the dataset name
    :param train_data_type: In this if it is true then we have train data else test data
    :return: We return the bag for words representation for spam and ham files
    """
    spam_file, ham_file, total_data, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset = import_train_data(
        dataset_name, train_data_type)
    total_file_dictionary = {}
    total_file_data = re.findall("[a-zA-Z]+", total_data)
    text_in_all_document = {}
    # at first we find all the words in the given dataset and find the frequencies in the whole dataset
    for each_word in total_file_data:
        each_word = each_word.lower()
        if each_word in total_file_dictionary:
            continue
        else:
            # I got the list of stem words from the nltk library
            if each_word not in ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                                 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                                 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                                 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                                 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                                 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                                 "won't", 'wouldn', "wouldn't"]:
                total_file_dictionary[each_word] = 0
        if each_word in text_in_all_document:
            if each_word not in ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                                 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                                 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                                 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                                 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                                 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                                 "won't", 'wouldn', "wouldn't"]:
                text_in_all_document[each_word] = text_in_all_document[each_word] + 1
        else:
            if each_word not in ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                                 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                                 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                                 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                                 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                                 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                                 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                                 "won't", 'wouldn', "wouldn't"]:
                text_in_all_document[each_word] = 1
    # In the following steps we find the words in the spam dataset and create the bag of words
    spam_email_bag_of_words = []
    spam_mail_in_all_documents = {}
    # The frequencies of each words are stored in the case of bag of words.
    for each_spam_mail in spam_file:
        temp_dict = copy.deepcopy(total_file_dictionary)
        each_spam_mail1 = re.findall("[a-zA-Z]+", each_spam_mail)
        # Here we create the bag of words for each document and append it in a list
        for each_word in each_spam_mail1:
            each_word = each_word.lower()
            if each_word in temp_dict:
                temp_dict[each_word] = temp_dict[each_word] + 1
        # Here we store all the words in the spam dataset
        spam_mail_in_all_documents = Counter(spam_mail_in_all_documents) + Counter(temp_dict)
        spam_email_bag_of_words.append(temp_dict)
    # In the following steps we find all the ham words and add them in the bag of words
    ham_mail_in_all_documents = {}
    ham_email_bag_of_words = []
    for each_ham_mail in ham_file:
        # Here we create the bag of words for each document and append it in a list
        temp_dict = copy.deepcopy(total_file_dictionary)
        each_ham_mail1 = re.findall("[a-zA-Z]+", each_ham_mail)
        for each_word in each_ham_mail1:
            each_word = each_word.lower()
            if each_word in temp_dict:
                temp_dict[each_word] = temp_dict[each_word] + 1
        # Here we store all the words in the ham dataset
        ham_mail_in_all_documents = Counter(ham_mail_in_all_documents) + Counter(temp_dict)
        ham_email_bag_of_words.append(temp_dict)
    return spam_email_bag_of_words, ham_email_bag_of_words, text_in_all_document, spam_mail_in_all_documents, ham_mail_in_all_documents, size_of_total_dataset, size_of_spam_dataset, size_of_ham_dataset, total_file_dictionary
