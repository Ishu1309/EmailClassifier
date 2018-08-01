from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.feature import HashingTF

import sys
import logging
import os


def check_usage():
    """
    Check the usage of email spam detection utility.
    Raise an exception in case of missing arguments.
    :return: None
    """
    if len(sys.argv) != 4:
        raise Exception('Usage:spark-submit Assign.py local <spam_email_path> <non_spam_email_path> <query_email_path>')


def read_files(path):
    """
    Create text file RDD using sparkcontext's textFile method
    :param path: full path to file
    :return: Text file RDD
    """
    if not os.path.exists(path):
        raise Exception('File - {0}, not found.'.format(path))

    dist_file = sc.textFile(path)
    return dist_file


def create_label_point_datasets(term_freq, label, mail):
    """
    Create label point datasets for spam and non-spam emails
    :param term_freq:
    :param label:
    :param mail:
    :return:
    """
    feature = mail.map(lambda email: term_freq.transform(email.split(' ')))
    return feature.map(lambda email_features: LabeledPoint(label, email_features))


def build_logistic_model(spam_data, non_spam_data):
    """
    Build a logistic regression model
    :param spam_data:
    :param non_spam_data:
    :return:
    """
    # Create a HashingTF instance with number of features = 100
    term_freq = HashingTF(numFeatures=100)

    positive_case = create_label_point_datasets(term_freq, 1, spam_data)
    negative_case = create_label_point_datasets(term_freq, 0, non_spam_data)

    # Combine positive and negative datasets and cache training data
    training_data = positive_case.union(negative_case)
    training_data.cache()

    # Run LogisticRegression
    logistic_model = LogisticRegressionWithLBFGS.train(training_data)
    res = {'tf': term_freq,
           'model': logistic_model,
           'positive_instance': positive_case,
           'negative_instance': negative_case}
    return res


def query_on_test_email(query_data):
    """
    Predict if an email is benign or spam based on logistic model
    :param query_data:
    :return:
    """
    query_predict = query_data.map(lambda email: (model.predict(tf.transform(email.split(" "))), email))
    main_logger.info(query_predict.collect())


def get_prediction_values(current_example):
    """

    :param current_example:
    :return:
    """
    test = current_example.map(lambda x: (model.predict(x.features), x.label))
    value = 1.0 * test.filter(lambda x: x[0] == x[1]).count()
    return value


if __name__ == "__main__":
    check_usage()

    # main logger logs details about main module
    main_logger = logging.getLogger('Email_Spam_Detection')

    # extract all command line arguments
    spam_path = sys.argv[2]
    non_spam_path = sys.argv[3]
    query_path = sys.argv[4]
    main_logger.info('spam_path = {0}'.format(spam_path))
    main_logger.info('non_spam_path = {0}'.format(non_spam_path))
    main_logger.info('query_path = {0}'.format(query_path))

    sc = SparkContext(sys.argv[1],"email_Spam_NoSpam")

    # Reading the spam,noSpam and query textFile
    spam_data = read_files(spam_path)
    non_spam_data = read_files(non_spam_path)
    query_data = read_files(query_path)

    result = build_logistic_model(spam_data, non_spam_path)
    model = result['model']
    tf = result['tf']

    query_on_test_email(query_data)

    # Spam and Non Spam predictions

    positive_prediction = get_prediction_values(result['positive_instance'])
    accuracy_positive = positive_prediction/spam_data.count()
    main_logger.info("accuracy_positive: {0}".format(accuracy_positive))

    negative_prediction = get_prediction_values(result['negative_instance'])
    accuracy_negative = negative_prediction / non_spam_data.count()
    main_logger.info("accuracy_negative: {0}".format(accuracy_negative))

    overall_accuracy = (positive_prediction + negative_prediction)/(spam_data.count() + non_spam_data.count())
    main_logger.info("overall_accuracy: {0}".format(overall_accuracy))

