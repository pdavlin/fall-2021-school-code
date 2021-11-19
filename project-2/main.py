from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

with open('./dataset-for-Project2.csv', 'r') as dataset_file:
    lines = dataset_file.readlines()
    dataset_file.close()

# Parse the dataset into a list of vectors
vectors = []
labels = []
for line in lines:
    line = line.strip()
    line_vals = line.split(',')
    vectors.append(line_vals[1:-1])
    labels.append(line_vals[-1])
# print(labels)

word_names = vectors[0]

# Remove the first line of IDs from the dataset
vectors = vectors[1:]
labels = labels[1:]

def test_classifier(newsgroup_train, newsgroup_test, classifier, labels):

    labels_train = labels[:11175]
    labels_test = labels[11175:]
    classifier.fit(newsgroup_train, labels_train)
    predictions = classifier.predict(newsgroup_test)
    accuracy = classifier.score(newsgroup_test, labels_test)


    cfm = confusion_matrix(labels_test, predictions)
    false_positives = cfm[0][1]
    false_negatives = cfm[1][0]
    print('accuracy: {}. {} false positives, {} false negatives'.format(accuracy, false_positives, false_negatives))

def calc_gini_index(data, labels):
    _, features = data.shape
    gini = np.ones(features) * 0.5
    # print(features)
    for i in range(features):
        v = np.unique(data[:,i])
        for j in range(len(v)):
            left_docs = labels[data[:,i] <= v[j]]
            right_docs = labels[data[:, i] > v[j]]
            
            gini_left = 0
            gini_right = 0

            for k in range(np.min(labels), np.max(labels)+1):
                if len(left_docs) != 0:
                    t1_left = np.true_divide(len(left_docs[left_docs == k]), len(left_docs))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left
                
                if len(right_docs) != 0:
                    t1_right = np.true_divide(len(right_docs[right_docs == k]), len(right_docs))
                    t2_right = np.power(t1_right, 2)
                    gini_left += t2_right
            gini_left = 1 - gini_left
            gini_right = 1 - gini_right

            t1_gini = (len(left_docs) * gini_right + len(right_docs) * gini_right)

            value = np.true_divide(t1_gini, len(labels))

            if value < gini[i]:
                gini[i]=value
            
    return gini

def convert_labels_to_int_array(labels):
    label_ints = []
    for i in range(len(labels)):
        label_ints.append(i)
    # print(label_ints)
    return label_ints

newsgroup_train = vectors[:11175]
newsgroup_test = vectors[11175:]
np_train = np.asarray(newsgroup_train)
# print(np_vectors.shape)
int_labels = convert_labels_to_int_array([i for i in range(np_train.shape[0])])
int_labels = np.asarray(int_labels)
print(len(int_labels))
gini = calc_gini_index(np_train, int_labels)
print(gini)


# classifier = RandomForestClassifier(max_depth=2)
# test_classifier(newsgroup_train, newsgroup_test, classifier, vectors, labels)

# classifier = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=95)
# test_classifier(classifier, vectors, labels)


