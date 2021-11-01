import os
import re

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from numpy import zeros, ones

print('.....PREPROCESSING.....')

filenums = [74,75,76,77,78,79]
doc_count = 1
docs_processed = 0

for num in filenums:
    filename = os.path.join('./input', 'cf{}.xml'.format(num))
    with open(filename, "r") as fp:
        lines = fp.readlines()

    with open(filename, "w") as fp:
        for line in lines:
            # for some reason this particular tag blows up the XML parser
            if line.strip("\n") != "<></>":
                fp.write(line)
    
    with open(filename, "r") as f:
        soup = BeautifulSoup(f, "xml")
        # get all the XML tags that correspond to article text
        contents = soup.find_all(["ABSTRACT", "EXTRACT"])
        docs_processed = docs_processed + len(contents)
        print('{} documents parsed'.format(docs_processed))
        for doc_text in contents:
            raw = doc_text.get_text().replace('\n', ' ')
            words = raw.split()
            stop_words = ENGLISH_STOP_WORDS
            # remove stop words if they exist
            words = [word.lower() for word in words if word not in stop_words and len(word) > 2]
            output_text = re.sub(r'[^\w\s]', '', ' '.join(words))
            with open('./output/{}.txt'.format(doc_count), 'w') as fw:
                fw.write(output_text)
            doc_count = doc_count + 1
print('Generating term set for corpus...')
term_set = []
for input_doc_id in range(1,1240):
    filename = os.path.join('./output', '{}.txt'.format(input_doc_id))

    with open(filename, "r") as fp:
        line = fp.readlines()[0]
        fp.close()
    words = line.split()
    for word in words:
        if word not in term_set:
            term_set.append(word)
print('{} terms in corpus'.format(len(term_set)))


print('.....PART 1.....')

def jaccard_similarity(input, comparison):
    a = 0
    b = 0
    c = 0

    for i in range(len(input)):
        if input[i] == 1 and comparison[i] == 1:
            a = a + 1
        elif input[i] == 1 and comparison[i] == 0:
            b = b + 1
        elif input[i] == 0 and comparison[i] == 1:
            c = c + 1
    
    denominator = a + b + c
    return a / denominator

print('Calculating boolean vectors for all documents...')
boolean_vecs = []
for doc_id in range(1, 1240):
    vec = zeros(len(term_set))
    filename = os.path.join('./output', '{}.txt'.format(doc_id))
    with open(filename, "r") as fp:
        line = fp.readlines()[0]
        fp.close()
    words = line.split()
    for word in words:
        if word in term_set:
            match_index = term_set.index(word)
            vec[match_index] = 1
    boolean_vecs.append(vec)

print('Calculating results for part 1...')
input_doc_ids = [1,2,3]
for input_doc_id in input_doc_ids:
    print('Comparing doc {} vector to set...'.format(input_doc_id))

    jaccard_results = {}
    for doc_id in range(1,1240):
        jaccard_results[doc_id] = jaccard_similarity(boolean_vecs[input_doc_id-1], boolean_vecs[doc_id-1])

    sorted_jaccard_keys = list(dict(sorted(jaccard_results.items(), key=lambda x:x[1])[-4:]).keys())[:3]
    sorted_jaccard_keys.reverse()
    print('Three most similar documents to doc {}: {}'.format(input_doc_id,sorted_jaccard_keys))


print('.....PART 2.....')

print('Calculating count vectors for all documents...')
count_vecs = []
for doc_id in range(1, 1240):
    # print(doc_id)
    vec = zeros(len(term_set))
    filename = os.path.join('./output', '{}.txt'.format(doc_id))
    with open(filename, "r") as fp:
        line = fp.readlines()[0]
        fp.close()
    words = line.split()
    for word in words:
        if word in term_set:
            match_index = term_set.index(word)
            vec[match_index] = vec[match_index] + 1
    count_vecs.append(vec)

print('Calculating results for part 2...')
cosine_array = cosine_similarity(count_vecs, count_vecs)
input_doc_ids = [1,2,3]
for input_doc_id in input_doc_ids:
    dict_key = 1
    cosine_result_dict = {}
    for i in cosine_array[input_doc_id - 1]:
        cosine_result_dict[dict_key] = i;
        dict_key = dict_key + 1
    sorted_cosine_keys = list(dict(sorted(cosine_result_dict.items(), key=lambda x:x[1])[-4:]).keys())[:3]
    sorted_cosine_keys.reverse()
    print('Three most similar documents to doc {}: {}'.format(input_doc_id,sorted_cosine_keys))

print('.....PART 3.....')

print('Iterating over term set to get total word count...')
idf = zeros(len(term_set))
for i in range(0,1239):
    # print(i)
    for j in range(len(count_vecs[i])):
        # print(j, idf[j])
        idf[j] = idf[j] + j
for i in range(len(idf)):
    if idf[i] != 0:
        idf[i] = 1239 / idf[i]

print('Calculating TFIDF vectors for all documents...')
tfidf_vecs = []
for doc_id in range(1, 1240):
    # print(doc_id)
    vec = zeros(len(term_set))
    filename = os.path.join('./output', '{}.txt'.format(doc_id))
    with open(filename, "r") as fp:
        line = fp.readlines()[0]
        word_count = len(line.split(' '))
        fp.close()
    words = line.split()
    for word in words:
        if word in term_set:
            match_index = term_set.index(word)
            vec[match_index] = vec[match_index] + 1
    for i in range(len(vec)):
        tf = vec[i]/word_count
        tfidf = tf * idf[i]
        vec[i] = tfidf
    tfidf_vecs.append(vec)

print('Calculating results for part 3...')
cosine_array = cosine_similarity(tfidf_vecs, tfidf_vecs)
input_doc_ids = [1,2,3]
for input_doc_id in input_doc_ids:
    dict_key = 1
    cosine_result_dict = {}
    for i in cosine_array[input_doc_id - 1]:
        cosine_result_dict[dict_key] = i;
        dict_key = dict_key + 1
    sorted_cosine_keys = list(dict(sorted(cosine_result_dict.items(), key=lambda x:x[1])[-4:]).keys())[:3]
    sorted_cosine_keys.reverse()
    print('Three most similar documents to doc {}: {}'.format(input_doc_id,sorted_cosine_keys))
