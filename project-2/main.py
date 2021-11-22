from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import random

with open('./dataset-for-Project2.csv', 'r') as dataset_file:
    lines = dataset_file.readlines()
    dataset_file.close()

# Parse the dataset into a list of vectors
vectors = []
labels = []
labels_unique = []
for line in lines:
    line = line.strip()
    line_vals = line.split(',')
    vectors.append(line_vals[1:-1])
    label = line_vals[-1]
    labels.append(label)
    if label not in labels_unique:
        labels_unique.append(label)

# Remove the first line of IDs from the dataset
vectors = vectors[1:]
labels = labels[1:]
labels_unique = labels_unique[1:]


# convert the labels to integers for use in the classifier and feature selection
# and save the labels in a dictionary for later use if needed
labels_dict = {}
for i in range(len(labels_unique)):
    labels_dict[labels_unique[i]] = i

labels_as_ints = []
for i in range(len(labels)):
    labels_as_ints.append(labels_dict[labels[i]])

# pick a random seed for the classifier to use across all tests
classifier_seed = random.randint(0, 100)
    

def test_classifier(data_train, data_test, clf, labels_train, labels_test):
    # fit the classifier and get results
    clf.fit(data_train, labels_train)
    predictions = clf.predict(data_test)

    # calculate the accuracy of the classifier on the test set
    accuracy = clf.score(data_test, labels_test)

    # get false positive and false negative results
    cfm = confusion_matrix(labels_test, predictions)
    false_positives = cfm[0][1]
    false_negatives = cfm[1][0]
    print('accuracy: {}. {} false positives, {} false negatives'.format(accuracy, false_positives, false_negatives))

# generate three classifier tests to use for all runs
def three_classifier_tests(data_train, data_test, labels_train, labels_test):
    print('Testing classifier results (seed {})...'.format(classifier_seed))

    print('Default settings')
    classifier = RandomForestClassifier(n_jobs = -1, random_state=classifier_seed)
    test_classifier(data_train, data_test, classifier, labels_train, labels_test)
    
    print('max_depth = 5')
    classifier = RandomForestClassifier(n_jobs = -1, max_depth = 5, random_state=classifier_seed)
    test_classifier(data_train, data_test, classifier, labels_train, labels_test)

    print('50% max_samples')
    classifier = RandomForestClassifier(max_samples=0.5, max_leaf_nodes=5, n_jobs = -1, random_state=classifier_seed)
    test_classifier(data_train, data_test, classifier, labels_train, labels_test)

# reduce dataset to only the features with the highest weights
def get_top_fifty_features(weights, train, test):
    proc = np.asarray(weights, dtype=float)
    indices = np.argsort(proc)
    indices = indices[-50:]
    return train[:, indices], test[:, indices]

# split data
newsgroup_train = vectors[:11175]
newsgroup_test = vectors[11175:]
labels_train = labels[:11175]
labels_test = labels[11175:]
labels_as_ints_train = labels_as_ints[:11175]
labels_as_ints_test = labels_as_ints[11175:]

np_train = np.asarray(newsgroup_train)
np_test = np.asarray(newsgroup_test)

print('=====DEFAULT=====')
three_classifier_tests(np_train, np_test, labels_train, labels_test)

print('=====GINI INDEX=====')

def calc_gini_index(data, labels):
    _, features = data.shape
    gini = np.ones(features)
    for i in range(features):
        v = np.unique(data[:,i])
        for j in range(len(v)):
            left_y = labels[data[:,i] <= v[j]]
            right_y = labels[data[:, i] > v[j]]
            
            g_left = 0
            g_right = 0

            for k in range(np.min(labels), np.max(labels)+1):
                if len(left_y) != 0:
                    # use True Divide to avoid division by zero
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    # square the result
                    t2_left = np.power(t1_left, 2)
                    # add result to the running summation
                    g_left += t2_left
                
                if len(right_y) != 0:
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    g_right += t2_right
            g_left = 1 - g_left
            g_right = 1 - g_right

            t1_gini = (len(left_y) * g_left + len(right_y) * g_right)

            value = np.true_divide(t1_gini, len(labels))

            if value < gini[i]:
                gini[i]=value
            
    return gini


gini_result = calc_gini_index(np_train, np.asarray(labels_as_ints_train))
train_top, test_top = get_top_fifty_features(gini_result, np_train, np_test)
three_classifier_tests(np.asarray(train_top), np.asarray(test_top), labels_train, labels_test)


print('=====CONDITIONAL ENTROPY=====')

def calc_conditional_entropy(data, labels):
    _ , features = data.shape
    results = np.zeros(features)
    for i in range(features):
        tj = labels[data[:,i] > 0]
        not_tj = labels[data[:,i] <= 0]
        result = 0

        for k in range(np.min(labels), np.max(labels)+1):

            # get a list of all the values in the column that match the current label
            cr = tj[tj == k]
            # and another list of the values that don't match the current label
            c_not_r = not_tj[not_tj == k]
            if len(cr > 0):
                # calculate P(c_r|tj)
                pcrtj = len(cr) / len(tj)
                if pcrtj > 0:
                    # calculate log(P(c_r|tj))
                    pcrtj = pcrtj * math.log10(pcrtj)
                    # calculate n(t_j)/n
                    n_over_n = len(cr)/len(labels)
                    pcrtj = pcrtj * n_over_n
            if len(c_not_r > 0):
                # do the same as above for the other half of the equation
                pcr_not_tj = len(c_not_r) / len(not_tj)
                if pcr_not_tj > 0:
                    pcr_not_tj = pcr_not_tj * math.log10(pcr_not_tj)
                    n_not_over_n = len(c_not_r)/len(labels)
                    pcr_not_tj = pcr_not_tj * n_not_over_n
            # result is negated summation
            result = result - (pcrtj + pcr_not_tj)
        results[i] = result
    return results

conditional_entropy_result = calc_conditional_entropy(np_train.astype(float), np.asarray(labels_as_ints_train))
train_top, test_top = get_top_fifty_features(conditional_entropy_result, np_train, np_test)
three_classifier_tests(np.asarray(train_top), np.asarray(test_top), labels_train, labels_test)


print('=====POINTWISE MUTUAL INFORMATION=====')

def calc_pointwise_mutual_information(data, labels):
    _ , features = data.shape
    results = np.zeros(features)
    for i in range(features):
        tj = labels[data[:,i] > 0]
        # I chose to use the max instead of the average, the books says it can be done either way
        # and I don't have time to do both.
        pmi_max = 0
        for k in range(np.min(labels), np.max(labels)+1):
            c_r = tj[tj == k]
            labels_matching = labels[labels == k]
            pcrtj = len(c_r) / len(tj)
            pcr = len(labels_matching) / len(labels)
            pcrtj_div_pcr = pcrtj / pcr
            pmi_r = 0
            if (pcrtj_div_pcr > 0):
                pmi_r = math.log10(pcrtj_div_pcr)
            # take the max of the pmi values for each
            if pmi_r > pmi_max:
                pmi_max = pmi_r
        results[i] = pmi_max
    return results

pointwise_result = calc_pointwise_mutual_information(np_train.astype(float), np.asarray(labels_as_ints_train))
train_top, test_top = get_top_fifty_features(pointwise_result, np_train, np_test)
three_classifier_tests(np.asarray(train_top), np.asarray(test_top), labels_train, labels_test)