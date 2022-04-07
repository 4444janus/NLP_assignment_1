# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
from importlib.resources import path
from nbformat import read
from sklearn import preprocessing

from collections import Counter
import spacy
from collections import Counter



# This loads a small English model trained on web data.
# For other models and languages check: https://spacy.io/models
nlp = spacy.load('en_core_web_sm')


#file_name = r"C:\Users\janus\OneDrive\Msc Artificial Intelligence\Jaar 1\P5\NLP\assignments\intro2nlp_assignment1_code\intro2nlp_assignment1_code\data\preprocessed\train\sentences.txt"
file = f"data/preprocessed/train/sentences.txt"
with open(file, "r", encoding="utf8") as in_file:
    data = in_file.read().rstrip().replace('\n',' ')
    
# Let's run the NLP pipeline on our test input
doc = nlp(data)


"""
1. Tokenization

"""

word_frequencies = Counter()
num_sentences = 0

# count word frequencies
for sentence in doc.sents:
    words = []
    # add sentence
    num_sentences += 1
    for token in sentence: 
        #filter punctuation
        if not token.is_punct:
            words.append(token.text)
    word_frequencies.update(words)

# get total tokens, words, types
num_tokens = len(doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())

# calculate average sentence length
average_words = round(num_words / num_sentences, 2)

# calculate average word length
letters = 0
for word in word_frequencies:
    # add length * the frequency of all words in text
    letters += word_frequencies[word] * len(word)

# devide total letters in text by total words
word_length = round(letters / num_words, 2)

#print results
#print(f"Tokens : {num_tokens}\nWords: {num_words}\nTypes: {num_types}\n")
#print(f"Sentences: {num_sentences}")
#print(f"Average per sentence: {average_words}")
#print(f"Average word length: {word_length}")

# Word definition: separate I'm into I & 'm. Keep words with "-" as one. Same words in different case are counted as different words.


"""
2. Word Classes

"""

most_common = word_frequencies.most_common(11)
#for i in most_common:
    #print(f"{i[0]} : {i[1]}")
# TODO fix case type

def count(doc):
    
    frequency_tag = Counter()
    frequency_pos = Counter()
    fine = []
    uni = []

    for token in doc:
        if not token.is_punct:
            fine.append(token.tag_)
            uni.append(token.pos_)
    frequency_tag.update(fine)
    frequency_pos.update(uni)

    return frequency_tag, frequency_pos

tag, pos = count(doc)
most_common_tag = tag.most_common(10)
most_common_pos = pos.most_common(10)

uni_pos = []
for j in most_common_pos:
    uni_pos.append(j[0])


def linked(tag):

    frequency = Counter()
    word = []

    for token in doc:
        if not token.is_punct:
            if token.tag_ == tag:
                word.append(token.text)
    frequency.update(word)

    return frequency.most_common(3), frequency.most_common()[-1]

# print table
for i in most_common_tag:
    uni = uni_pos[most_common_tag.index(i)]
    link, least = linked(i[0])
    words = []
    for word in link:
        words.append(word[0])
    relative = round(i[1] / num_types,2)
    #print(f"{i[0]}, {uni}, {i[1]}, {relative}, {words}, {least[0]}")

"""
3. N-Grams

"""
from nltk import ngrams

# When I use doc as input instead of the "raw" data as input it gives different results
# the 'raw' data seems to work, once the double backslash is filtered.

tokens = list(filter(('\\"').__ne__, data.split())) # Take the raw data and filter out \\

# Get a list of all the POS
def POS(doc):
    pos = []
    for token in doc:
        if not token.is_punct:
            pos.append(token.pos_)
    return pos

# Use the NLTK ngrams function and then count the most common n-grams
def NGrams(input, n):
    return Counter(ngrams(input, n)).most_common(3)

print("Token bigram:" , NGrams(tokens, 2))
print("Token trigram:", NGrams(tokens, 3))

pos = POS(doc)

print("POS bigram:"   , NGrams(pos, 2))
print("POS trigram:"  , NGrams(pos, 3))

"""
4. Lemmatization

"""   
lemma_triples = [] # First element is the lemma, second element is the infliction
                   # Third the sentence

for sentence in doc.sents:
    for token in sentence: 
        #filter punctuation
        if not token.is_punct and not token.text == token.lemma_ and not (token.lemma_ == '-PRON-'):
            # I found the example of abuse:
            if token.lemma_ == 'abuse':
                lemma_triples.append((token.lemma_, token.text, sentence))

sorted_triple = sorted(lemma_triples, key=lambda tup: tup[0])
for s in sorted_triple:
    print(s, '\n')

#print(sorted(lemma_tuples, key=lambda tup: tup[1]))

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





# assignment_1()
#assignment_2()
