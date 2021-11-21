from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

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
# labels_unique_int = [i for i in range(len(labels_unique))]
# print(labels_unique_int)

    

def test_classifier(data_train, data_test, clf, labels_train, labels_test):
    print(data_train.shape)
    print(data_test.shape)
    clf.fit(data_train, labels_train)
    predictions = clf.predict(data_test)
    accuracy = clf.score(data_test, labels_test)


    cfm = confusion_matrix(labels_test, predictions)
    false_positives = cfm[0][1]
    false_negatives = cfm[1][0]
    print('accuracy: {}. {} false positives, {} false negatives'.format(accuracy, false_positives, false_negatives))

def three_classifier_tests(data_train, data_test, labels_train, labels_test):
    classifier = RandomForestClassifier()
    test_classifier(data_train, data_test, classifier, labels_train, labels_test)
    
    classifier = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=95)
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

newsgroup_train = vectors[:11175]
newsgroup_test = vectors[11175:]
labels_train = labels[:11175]
labels_test = labels[11175:]
labels_as_ints_train = labels_as_ints[:11175]
labels_as_ints_test = labels_as_ints[11175:]

print('=====DEFAULT=====')
# three_classifier_tests(newsgroup_train, newsgroup_test, labels_train, labels_test)

print('=====GINI INDEX=====')
print('getting top fifty features...')
np_train = np.asarray(newsgroup_train)
# gini = calc_gini_index(np_train, np.asarray(labels_as_ints_train))
# top_fifty_features = get_top_fifty_features(gini)
# print('top fifty features: {}'.format(top_fifty_features))

# train_top_fifty_only = []
# for i in newsgroup_train:
#     vector_top_fifty = []
#     for j in range(len(i)):
#         if j in top_fifty_features:
#             vector_top_fifty.append(j)
#     train_top_fifty_only.append(vector_top_fifty)


# test_top_fifty_only = []
# for i in newsgroup_test:
#     vector_top_fifty = []
#     for j in range(len(i)):
#         if j in top_fifty_features:
#             vector_top_fifty.append(j)
#     test_top_fifty_only.append(vector_top_fifty)

# three_classifier_tests(np.asarray(train_top_fifty_only), np.asarray(newsgroup_test), labels_train, labels_test)


print('=====CONDITIONAL ENTROPY=====')

def calc_conditional_entropy(data, terms, labels):
    print(len(data), len(terms))
    for i in range(len(terms)):
        # Get number of documents in the dataset that contain the term (e.g., value > 0)
        n = 0
        for j in range(len(data)):
            if data[j][i] > 0:
                n = n + 1
        # number of docs not containing the term
        not_n = len(data) - n
        p_t = 0
        p_not_t = 0
        for k in range(len(labels)):
            noop
        log_p_t = np.log(p_t)
        log_p_not_t = np.log(p_t)
        # Get fraction of documents pertaining to label
        # Get 

calc_conditional_entropy(np_train.astype(float), np.asarray(word_names), np.asarray(labels))


print('=====POINTWISE MUTUAL INFORMATION=====')

