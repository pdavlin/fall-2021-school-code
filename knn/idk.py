from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

newsgroups_all = fetch_20newsgroups(subset='all', remove=(
    'headers', 'footers'), random_state=42, shuffle=True)
vectorizer = TfidfVectorizer()


X = vectorizer.fit_transform(newsgroups_all.data)
y = newsgroups_all.target

for test_size in [0.6, 0.4, 0.2, 0.1]:
    print('results for k-Values in test size {}'.format(test_size))
    for k_value in range(1, 21):
        sum = 0
        for _ in range(0, 10):
            classifier = KNeighborsClassifier(n_neighbors=k_value)

            newsgroups_train, newsgroups_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)
            classifier.fit(newsgroups_train, y_train)

            prediction = classifier.predict(newsgroups_test)

            sum = sum + accuracy_score(y_test, prediction)
        print('{}, {}'.format(k_value, sum/10))
