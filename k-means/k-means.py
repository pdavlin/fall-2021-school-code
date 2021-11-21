from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
import matplotlib.pyplot as plt
import pandas as pd
import re
# importing a `KMeans` library feels like skipping a bunch of steps but I think the goal of the assignment
# is to build my understanding of how the algorithm works in practice, as a function of the number
# of clusters, and not to spend a lot of time writing a bunch of lines to implement it.
from sklearn.cluster import KMeans

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers'), random_state=42, shuffle=True)

for i in range(len(newsgroups_train.data)):
    txt = re.sub(r'[^a-zA-Z]+', ' ', newsgroups_train.data[i])
    # get words from text
    words = txt.split()
    # remove stop words if they exist
    words = [word.lower() for word in words if word not in stopwords and len(word) > 2]
    # rejoin words, less any stop words
    newsgroups_train.data[i] = ' '.join(words)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(newsgroups_train.data)

clusters = [5,10,20,30]

for cluster_num in clusters:
    k_means_output = KMeans(n_clusters=cluster_num, max_iter=500).fit(tfidf)
    numcom = 2

    features = NMF(n_components=numcom).fit_transform(tfidf)
    plt.scatter(features[:,0], features[:,1], c = k_means_output.labels_, marker='+')
    plt.title('NMF - {} clusters'.format(cluster_num))
    plt.show()

    features = TruncatedSVD(n_components=numcom).fit_transform(tfidf)
    plt.scatter(features[:,0], features[:,1], c = k_means_output.labels_, marker='+')
    plt.title('TruncatedSVD - {} clusters'.format(cluster_num))
    plt.show()