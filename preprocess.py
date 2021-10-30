import os

from bs4 import BeautifulSoup

print('.....PREPROCESSING.....')

filenums = [74,75,76,77,78,79]
# filenums = [74]
doc_count = 1
docs_processed = 0

for num in filenums:
    filename = os.path.join('./input', 'cf{}.xml'.format(num))
    # with open(filename, "r") as fp:
    #     lines = fp.readlines()

    # with open(filename, "w") as fp:
    #     for line in lines:
    #         if line.strip("\n") != "<></>":
    #             fp.write(line)
    
    with open(filename, "r") as f:
        soup = BeautifulSoup(f, "xml")
        contents = soup.find_all(["ABSTRACT", "EXTRACT"])
        docs_processed = docs_processed + len(contents)
        print('{} documents parsed'.format(docs_processed))
        for text in contents:
            raw = text.get_text()
            with open('./output/{}.txt'.format(doc_count), 'w') as fw:
                fw.write(raw.replace('\n', ' '))
            doc_count = doc_count + 1

print('.....PART 1.....')