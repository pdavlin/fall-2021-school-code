Project 1
a. Your name:
    Patrick Davlin

b. Commands for executing your code.
    pip install bs4 sklearn numpy lxml
    python3 project-1.py
    (note: I used python version 3.8 for this assignment. Functions related to sorting dictionary
    objects require python 3.6 or above.)

c. A small paragraph comparing the results from the three methods. What are the 
similarities and differences in the output from the parts?

Results were as follows, ranked from most similar to third-most similar document:
.....PART 1.....
Three most similar documents to doc 1: [415, 987, 989]
Three most similar documents to doc 2: [345, 170, 1023]
Three most similar documents to doc 3: [4, 639, 113]
.....PART 2.....
Three most similar documents to doc 1: [415, 160, 778]
Three most similar documents to doc 2: [170, 324, 406]
Three most similar documents to doc 3: [4, 231, 293]
.....PART 3.....
Three most similar documents to doc 1: [415, 778, 160]
Three most similar documents to doc 2: [170, 324, 51]
Three most similar documents to doc 3: [4, 231, 293]

The results were fairly consistent across each method. Document 1's most similar similar document
was 415 and Document 3's most similar document was 4. Document 170 was similar to Document 2 in all
cases. Seeing the same documents appear consistently across all the specified methods is an 
encouraging sign that the algorithms were implemented properly.

d. What is your explanation for these simularities and differences?

The algorithms differ based on how they weight individual terms in a term set. In Part 1, the
algorithm only weights based on whether a word appears at all. In Part 2, terms that appear the most
in each compared document are weighted highly. In Part 3, terms that appear most frequently in the 
compared documents, but *less* frequently overall, are weighted highly. This is an interesting shift;
it indicates that documents which appear more often in comparison (document 4, when compared to 
document 3, for example) contain enough similarity that the weighting does not matter. Moreover,
it could be reasonably concluded that Documents 3 and 4 would be clustered closely since they contain
a number of similar, unique terms. Documents that change between iterations (for example, Document 2
has fairly few common similar Documents across the three algorithms) may have fewer terms in common
with other documents, and those documents may overlap more significantly with other documents across
the corpus (in other words, they may be less unique). Weighting the contents of Document 2 in a count 
vector made a significant difference compared to the boolean vectors, suggesting Document 2 uses many
terms (making it similar to Document 345, for example) but also uses some terms very frequently 
(making it similar to document 324).

e. What was the most challenging part of implementing the project?

I have never worked with XML before (since it has generally been superseded in new applications by
newer standards like JSON and YAML), and the process of finding a library that could effectively 
parse the XML documents to extract text proved difficult. Fortunately, the imported BeautifulSoup
(bs4) library handles XML parsing fairly efficiently.

f. What is your “takeaway” from the project?

This project indicates to me that there are several criteria that can define similarity, and that
changing those critera can significantly impact the results of clustering and similarity in, say, a
search engine. None of these individual results seems necessarily "better" or "more correct" in
context; it was interesting to see how they might be applied in the real world.
