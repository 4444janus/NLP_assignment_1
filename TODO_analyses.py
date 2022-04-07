# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy



# This loads a small English model trained on web data.
# For other models and languages check: https://spacy.io/models
nlp = spacy.load('en_core_web_sm')

file_name = r"C:\Users\janus\OneDrive\Msc Artificial Intelligence\Jaar 1\P5\NLP\assignments\intro2nlp_assignment1_code\intro2nlp_assignment1_code\data\preprocessed\train\sentences.txt"
data = open(file_name, encoding="utf8").read()
# Let's run the NLP pipeline on our test input
doc = nlp(data)

#########print tokenization
# for token in doc:
#     print(token.i, token, token.idx)
# print()

##########sentence segmentation
# sentences = doc.sents
# for sentence in sentences:
#     print()
#     print(sentence)
#     for token in sentence:
#         print(token.text)
#
# for token in doc:
#     print(token.text, token.lemma_)

#######pos tagging
# for token in doc:
#     print(token.text, token.pos_, token.tag_)


###########lemmatization
# for token in doc:
#     print(token.text, token.lemma_)

#########named entity
# for ent in doc.ents:
#     print(ent.text, ent.label_)

############display - not working
# from spacy import displacy
# displacy.render(doc, jupyter=False, style='ent')

#############token frequencies


# Our test input is the first paragraph of https://spacy.io/usage/linguistic-features
# Let's run the NLP pipeline on our test input

from collections import Counter
word_frequencies = Counter()

for sentence in doc.sents:
    words = []
    for token in sentence:
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
    word_frequencies.update(words)

print(word_frequencies)

#############
num_tokens = len(doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())

# print(num_tokens, num_words, num_types)
# print(f"num_tokens: {num_tokens} num_words: {num_words} num_types: {num_types}")

#assignment
#PART A
#1
# Process the dataset using the spaCy package and extract the following information:
# Number of tokens:
# Number of types:
# Number of words:
# Average number of words per sentence:
# Average word length:
#
# Provide the definition that you used to determine words:

def assignment_1():
    print(f"num_tokens: {num_tokens} num_types: {num_types} num_words: {num_words}")

    average_words = 5
    print(f"Average number of words per sentence:{average_words} ")

def assignment_2():
    for token in doc:
        print(token.text, token.pos_, token.tag_)

# assignment_1()
assignment_2()