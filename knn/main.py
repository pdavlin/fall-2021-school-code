# Import necessary modules
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import re
 
# Loading data
 
def preprocess_newsgroup_data(newsgroup_data):
    for i in range(len(newsgroup_data.data)):
        txt = re.sub(r'[^a-zA-Z]+', ' ', newsgroup_data.data[i])
        # get words from text
        words = txt.split()
        stop_words = ENGLISH_STOP_WORDS
        # remove stop words if they exist
        words = [word.lower() for word in words if word not in stop_words and len(word) > 2]
        # rejoin words, less any stop words
        newsgroup_data.data[i] = ' '.join(words)
    print(newsgroup_data.target_names)

    print(len(newsgroup_data.data))
    return newsgroup_data

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers'), random_state=42, shuffle=True)
# newsgroups_train = preprocess_newsgroup_data(newsgroups_train)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(newsgroups_train.data)
# print(tfidf.shape)
knn = KNeighborsClassifier(n_neighbors=5)
classifier = knn.fit(tfidf, newsgroups_train.target)

newsgroups_test = fetch_20newsgroups(subset='test',
                               remove=('headers','footers'), shuffle=True, random_state=42)

# newsgroups_test = preprocess_newsgroup_data(newsgroups_test)

tfidf_test = vectorizer.fit_transform(newsgroups_test.data)
classifier.predict(tfidf_test)
# print(tfidf_test.shape)