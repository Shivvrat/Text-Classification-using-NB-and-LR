import sys
import warnings
import evaluation_metrics_MCAP
import evaluation_metrics_Multi_Nomial_NB
import evaluation_metrics_SGD
import evaluation_metrics_discrete_naive_bayes
warnings.filterwarnings("ignore")


# Here we take the inputs given in the command line
arguments = list(sys.argv)
try:
    data_set_name = str(arguments[1])
    algorithm_name = str(arguments[2])
except:
    print "You have not provided enough arguments, please read the readme file"
    exit(-1)
try:
    type_of_model = str(arguments[3])
except:
    print "You have either chosen multi-nomial naive bayes or discrete naive bayes, else you need to provide one more parameter. Please check the readme"
    print ""
# Please keep all the datasets folder in the same directory as the code

def main():
    """
    This is the main function which is used to run all the algorithms
    :return:
    """
    try:
        if algorithm_name == '-mnnb':
            # This is for multi-nomial naive bayes
            evaluation_metrics = evaluation_metrics_Multi_Nomial_NB.evaluate_multinomial_NB(data_set_name)
        elif algorithm_name == '-dnb':
            # This is for discrete naive bayes
            evaluation_metrics = evaluation_metrics_discrete_naive_bayes.evaluate_discrete_NB(data_set_name)
        elif algorithm_name == '-mcap':
            # This is for the MCAP algorithm
            if type_of_model == '-bow':
                # This is for the bag of words model
                evaluation_metrics = evaluation_metrics_MCAP.evaluate_MCAP_bag_of_words(data_set_name)
            elif type_of_model == '-bm':
                # This is for the bernoulli model
                evaluation_metrics = evaluation_metrics_MCAP.evaluate_MCAP_bernoulli_model(data_set_name)
            else:
                print 'You have entered wrong value for the type of model, please check.'
                return
        elif algorithm_name == '-sgd':
            # This is for the SGD classifier from sklearn
            if type_of_model == '-bow':
                # This is for the bag of words model
                evaluation_metrics = evaluation_metrics_SGD.evaluate_SGD_bag_of_words(data_set_name)
            elif type_of_model == '-bm':
                # This is for the bernoulli model
                evaluation_metrics = evaluation_metrics_SGD.evaluate_SGD_bernoulli_model(data_set_name)
            else:
                print 'You have entered wrong value for the type of model, please check.'
        else:
            print 'You have entered wrong value for the algorithm, please check in readme'
            return
        print "The accuracy is", evaluation_metrics[0]
        print "The Precision is", evaluation_metrics[1]
        print "The Recall is", evaluation_metrics[2]
        print "The F1 Score is", evaluation_metrics[3]
        try:
            value = evaluation_metrics[4]
            print "The selected value of lambda is", value
        except:
            i = 0
    except:
        print 'Something went wrong please check the command line parameters again from the readme or check if the dataset folder is in the right place or not'

if __name__ == "__main__":
    main()