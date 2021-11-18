from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

# Remove the first line of IDs from the dataset
vectors = vectors[1:]
labels = labels[1:]

# Split to train and test sets


# Confirm that the dataset is the correct size
# print(len(newsgroup_train),len(newsgroup_test))

# Print the first vector and labels
# print(len(newsgroup_train))
# print(len(labels_train))

def test_classifier(classifier, vectors, labels):
    newsgroup_train = vectors[:11175]
    newsgroup_test = vectors[11175:]

    labels_train = labels[:11175]
    labels_test = labels[11175:]
    classifier.fit(newsgroup_train, labels_train)
    predictions = classifier.predict(newsgroup_test)
    accuracy = classifier.score(newsgroup_test, labels_test)


    cfm = confusion_matrix(labels_test, predictions)
    false_positives = cfm[0][1]
    false_negatives = cfm[1][0]
    print('accuracy: {}. {} false positives, {} false negatives'.format(accuracy, false_positives, false_negatives))


classifier = RandomForestClassifier(max_depth=2)
test_classifier(classifier, vectors, labels)

classifier = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=95)
test_classifier(classifier, vectors, labels)


