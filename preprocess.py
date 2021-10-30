import os

from bs4 import BeautifulSoup

filenums = [74,75,76,77,78,79]
# filenums = [74]
doc_count = 0

for num in filenums:
    filename = os.path.join('./input', 'cf{}.xml'.format(num))
    with open(filename, "r") as fp:
        lines = fp.readlines()

    with open(filename, "w") as fp:
        for line in lines:
            if line.strip("\n") != "<></>":
                fp.write(line)
    
    # with open(filename, "r") as f:
    #     soup = BeautifulSoup(f)