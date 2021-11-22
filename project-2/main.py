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
# print(labels_unique)

word_names = vectors[0]

# Remove the first line of IDs from the dataset
vectors = vectors[1:]
labels = labels[1:]
labels_unique = labels_unique[1:]
labels_dict = {}
for i in range(len(labels_unique)):
    labels_dict[labels_unique[i]] = i
# print(labels_dict)

labels_as_ints = []
for i in range(len(labels)):
    labels_as_ints.append(labels_dict[labels[i]])

classifier_seed = random.randint(0, 100)
    

def test_classifier(data_train, data_test, clf, labels_train, labels_test):
    clf.fit(data_train, labels_train)
    predictions = clf.predict(data_test)
    accuracy = clf.score(data_test, labels_test)


    cfm = confusion_matrix(labels_test, predictions)
    false_positives = cfm[0][1]
    false_negatives = cfm[1][0]
    print('accuracy: {}. {} false positives, {} false negatives'.format(accuracy, false_positives, false_negatives))

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


def calc_gini_index(data, labels):
    _, features = data.shape
    gini = np.ones(features)
    # print(features)
    for i in range(features):
        v = np.unique(data[:,i])
        for j in range(len(v)):
            left_y = labels[data[:,i] <= v[j]]
            right_y = labels[data[:, i] > v[j]]
            
            gini_left = 0
            gini_right = 0

            for k in range(np.min(labels), np.max(labels)+1):
                if len(left_y) != 0:
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left
                
                if len(right_y) != 0:
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right
            gini_left = 1 - gini_left
            gini_right = 1 - gini_right

            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

            value = np.true_divide(t1_gini, len(labels))

            if value < gini[i]:
                gini[i]=value
            
    return gini

def convert_labels_to_int_array(labels):
    label_ints = []
    for i in range(len(labels)):
        label_ints.append(i)
    return label_ints

def get_top_fifty_features(weights):
    docs_dict = {}
    for i in range(len(weights)):
        docs_dict[weights[i]] = i
    docs_dict_sorted_keys = sorted(docs_dict)[:50]
    # print(docs_dict_sorted_keys)
    top_fifty_features = []
    for j in docs_dict_sorted_keys:
        top_fifty_features.append(docs_dict[j])
    return sorted(top_fifty_features)

def top_fifty_alt(weights):
    proc = np.asarray(weights, dtype=float)
    indices = np.argsort(proc)
    indices = indices[-50:]
    return indices

newsgroup_train = vectors[:11175]
newsgroup_test = vectors[11175:]
labels_train = labels[:11175]
labels_test = labels[11175:]
labels_as_ints_train = labels_as_ints[:11175]
labels_as_ints_test = labels_as_ints[11175:]

print('=====DEFAULT=====')
three_classifier_tests(newsgroup_train, newsgroup_test, labels_train, labels_test)

print('=====GINI INDEX=====')
np_train = np.asarray(newsgroup_train)
gini = calc_gini_index(np_train, np.asarray(labels_as_ints_train))
# top_fifty_features = get_top_fifty_features(gini)
top_fifty_features = top_fifty_alt(gini)
print('top fifty features: {}'.format(top_fifty_features))

train_top_fifty_only = np_train[:, top_fifty_features]
test_top_fifty_only = np.asarray(newsgroup_test)[:, top_fifty_features]


three_classifier_tests(np.asarray(train_top_fifty_only), np.asarray(test_top_fifty_only), labels_train, labels_test)


print('=====CONDITIONAL ENTROPY=====')

def calc_conditional_entropy(data, labels):
    _ , features = data.shape
    results = np.zeros(features)
    for i in range(features):
        tj = labels[data[:,i] > 0]
        not_tj = labels[data[:,i] <= 0]
        result = 0

        for k in range(np.min(labels), np.max(labels)+1):
            cr = tj[tj == k]
            c_not_r = not_tj[not_tj == k]
            if len(cr > 0):
                pcrtj = len(cr) / len(tj)
                if pcrtj > 0:
                    pcrtj = pcrtj * math.log10(pcrtj)
                    n_over_n = len(cr)/len(labels)
                    pcrtj = pcrtj * n_over_n
            if len(c_not_r > 0):
                pcr_not_tj = len(c_not_r) / len(not_tj)
                if pcr_not_tj > 0:
                    pcr_not_tj = pcr_not_tj * math.log10(pcr_not_tj)
                    n_not_over_n = len(c_not_r)/len(labels)
                    pcr_not_tj = pcr_not_tj * n_not_over_n
            result = result - (pcrtj + pcr_not_tj)
            # print(result)
        results[i] = result
    return results

conditional_entropy_result = calc_conditional_entropy(np_train.astype(float), np.asarray(labels_as_ints_train))
# print(conditional_entropy_result)
ce_top_fifty = top_fifty_alt(conditional_entropy_result)
print('top fifty features: {}'.format(ce_top_fifty))

train_top_fifty_only = np_train[:, ce_top_fifty]
test_top_fifty_only = np.asarray(newsgroup_test)[:, ce_top_fifty]

three_classifier_tests(np.asarray(train_top_fifty_only), np.asarray(test_top_fifty_only), labels_train, labels_test)


print('=====POINTWISE MUTUAL INFORMATION=====')

def calc_pointwise_mutual_information(data, labels):
    _ , features = data.shape
    results = np.zeros(features)
    for i in range(features):
        tj = labels[data[:,i] > 0]
        pmi_max = 0
        for k in range(np.min(labels), np.max(labels)+1):
            k_tj = tj[tj == k]
            k_total = labels[labels == k]
            pcrtj = len(k_tj) / len(tj)
            pcr = len(k_total) / len(labels)
            pcrtj_div_pcr = pcrtj / pcr
            pmi_r = 0
            if (pcrtj_div_pcr > 0):
                pmi_r = math.log10(pcrtj_div_pcr)
            if pmi_r > pmi_max:
                pmi_max = pmi_r
        results[i] = pmi_max
    return results

pw_result = calc_pointwise_mutual_information(np_train.astype(float), np.asarray(labels_as_ints_train))
pw_top_fifty = top_fifty_alt(pw_result)
print('top fifty features: {}'.format(pw_top_fifty))

train_top_fifty_only = np_train[:, pw_top_fifty]
test_top_fifty_only = np.asarray(newsgroup_test)[:, pw_top_fifty]

three_classifier_tests(np.asarray(train_top_fifty_only), np.asarray(test_top_fifty_only), labels_train, labels_test)