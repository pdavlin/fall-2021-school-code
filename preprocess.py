import os

from bs4 import BeautifulSoup

filenums = [74,75,76,77,78,79]
doc_count = 0

for num in filenums:
    filename = 'cf{}.xml'.format(num)
    with open(os.path.join('./input', filename)) as f:
        filelines = f.readlines()
        for line in filelines:
            if line.startswith('<RECORD>'):
                doc_count = doc_count + 1
            with open(os.path.join('./output', '{}.txt'.format(doc_count)), 'a') as o:
                o.write(line)
                o.close()
        f.close()