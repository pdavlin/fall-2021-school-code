import os
import re

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print('.....PREPROCESSING.....')

filenums = [74,75,76,77,78,79]
# filenums = [74]
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

print('.....PART 1.....')