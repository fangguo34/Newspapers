import svm
import logreg
import math
import util
import collections
import numpy as np
import re

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    # flattens a list of lists to a list
    words_list = []

    for sublist in messages:
        for item in sublist:
            words_list.append(item)

    # build dictionary
    dictionary = {}
    i = 0
    print(len(set(words_list)))

    for x in sorted(set(words_list)):
        chn = re.findall(r'[\u4e00-\u9fff]+', x)
        if (words_list.count(x) >=5) & (len(chn) > 0):
            dictionary[x] = i
            i += 1
            print(i)

    return(dictionary)
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of lists of words.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    m = len(messages)
    n = len(word_dictionary)
    print(n)
    matrix_output = np.zeros((m,n))

    for j, word in enumerate(word_dictionary):
        print(j)
        print(word)
        for i, msg in enumerate(messages): 
            matrix_output[i,j] = msg.count(word)

    return(matrix_output)

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of the fitted model, consisting of the 
    learned model parameters.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n = len(labels)
    n_1 = sum(labels == 1)
    v = matrix.shape[1]

    phi_y = n_1/n

    phi_1_num = np.sum(matrix[labels == 1,], axis = 0) + 1
    phi_1_deno = v + np.sum(np.sum(matrix[labels == 1,], axis = 1))
    phi_1 = phi_1_num/phi_1_deno

    phi_0_num = np.sum(matrix[labels == 0,], axis = 0) + 1
    phi_0_deno = v + np.sum(np.sum(matrix[labels == 0,], axis = 1))
    phi_0 = phi_0_num/phi_0_deno

    model = {'phi_y':phi_y, "phi_1":phi_1, "phi_0":phi_0}
    return(model)

    # *** END CODE HERE ***

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model (int array of 0 or 1 values)
    """
    # *** START CODE HERE ***
    phi_y,phi_1,phi_0=model["phi_y"],model["phi_1"],model["phi_0"]


    log_phi_1 = np.log(phi_1)
    prob_1 = np.exp(matrix.dot(log_phi_1) + np.log(phi_y))
    # print(prob_1.shape)

    log_phi_0 = np.log(phi_0)
    prob_0 = np.exp(matrix.dot(log_phi_0) + np.log(1-phi_y))
    # print(prob_0.shape)

    predict_prob = prob_1/(prob_1+prob_0)
    predict_spam = predict_prob>=0.5

    return(predict_spam)

    # *** END CODE HERE ***


def get_top_naive_bayes_words(model, dictionary, top):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    # If use skitlearn
    phi_y = np.exp(model.class_log_prior_)
    phi_0_1 = model.feature_log_prob_
    phi_0 = np.exp(phi_0_1[0,:])
    phi_1 = np.exp(phi_0_1[1,:])

    # If use my own
    # phi_y,phi_1,phi_0=model["phi_y"],model["phi_1"],model["phi_0"]

    # Findmost indicative words
    indicator = np.log(phi_1/phi_0)
    indicator_top_5 = indicator.argsort()[-top:][::-1]

    dict_array = np.array(list(dictionary.items()))
    dict_top_5 = [dict_array[index,0] for index in indicator_top_5]

    return(dict_top_5)

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use validation set accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracy = 0
    radius_opt = 0

    for radius in radius_to_consider:
        predicts = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy_new = np.mean(predicts == val_labels)
        if accuracy_new > accuracy:
            accuracy = accuracy_new
            radius_opt = radius

    return(radius_opt)



    # *** END CODE HERE ***

def compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, learning_rates_to_consider):
    """Compute the best logistic regression learning rate using the provided training and evaluation datasets.

    You should only consider learning rates within the learning_rates_to_consider list.
    You should use validation set accuracy as a metric for comparing the different learning rates.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        learning_rates_to_consider: The learning rates to consider

    Returns:
        The best logistic regression learning rate which maximizes validation set accuracy.
    """
    # *** START CODE HERE ***
    accuracy = 0
    lr_opt = 0 
    
    for lr in learning_rates_to_consider:
        predicts = logreg.train_and_predict_logreg(train_matrix, train_labels, val_matrix, lr)
        accuracy_new = np.mean(predicts == val_labels)
        if accuracy_new > accuracy:
            accuracy = accuracy_new
            lr_opt = lr

    return(lr_opt)



    # *** END CODE HERE ***


# def find_word_in_sentence(message, word):
#     return((message == word).sum())


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    # print(val_matrix.shape)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    # print(naive_bayes_model)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


    train_matrix = util.load_bert_encoding('bert_train_matrix.tsv.bz2')
    val_matrix = util.load_bert_encoding('bert_val_matrix.tsv.bz2')
    test_matrix = util.load_bert_encoding('bert_test_matrix.tsv.bz2')
    
    best_learning_rate = compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.001, 0.0001, 0.00001, 0.000001])

    print('The best learning rate for logistic regression is {}'.format(best_learning_rate))

    logreg_predictions = logreg.train_and_predict_logreg(train_matrix, train_labels, test_matrix, best_learning_rate)

    logreg_accuracy = np.mean(logreg_predictions == test_labels)

    print('The Logistic Regression model with BERT encodings had an accuracy of {} on the testing set'.format(logreg_accuracy))
    

if __name__ == "__main__":
    main()
