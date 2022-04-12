# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np

from importlib.resources import path
from nbformat import read
from numpy import median
from sklearn import preprocessing
from nltk import ngrams
from nltk.tokenize import word_tokenize
from wordfreq import zipf_frequency, word_frequency
from scipy import stats

from collections import Counter



"""
1. Tokenization

"""
def frequencies(doc):
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

    return word_frequencies, num_sentences

def tokenization(doc):

    word_frequencies, num_sentences = frequencies(doc)

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
    print(f"Tokens : {num_tokens}\nWords: {num_words}\nTypes: {num_types}\n")
    print(f"Sentences: {num_sentences}")
    print(f"Average per sentence: {average_words}")
    print(f"Average word length: {word_length}")

    # Word definition: separate I'm into I & 'm. Keep words with "-" as one. Same words in different case are counted as different words.

"""
2. Word Classes

"""
def word_classes(doc):
    word_frequencies, _ = frequencies(doc)
    most_common = word_frequencies.most_common(11)
    for i in most_common:
        print(f"{i[0]} : {i[1]}")
    # TODO fix case type

    tag, pos = count(doc)
    most_common_tag = tag.most_common(10)
    most_common_pos = pos.most_common(10)

    uni_pos = []
    for j in most_common_pos:
        uni_pos.append(j[0])

    # print table
    num_types = len(word_frequencies.keys())
    for i in most_common_tag:
        uni = uni_pos[most_common_tag.index(i)]
        link, least = linked(i[0])
        words = []
        for word in link:
            words.append(word[0])
        relative = round(i[1] / num_types,2)
        print(f"{i[0]}, {uni}, {i[1]}, {relative}, {words}, {least[0]}")


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


def linked(tag):

    frequency = Counter()
    word = []

    for token in doc:
        if not token.is_punct:
            if token.tag_ == tag:
                word.append(token.text)
    frequency.update(word)

    return frequency.most_common(3), frequency.most_common()[-1]


"""
3. N-Grams

"""

# When I use doc as input instead of the "raw" data as input it gives different results
# the 'raw' data seems to work, once the double backslash is filtered.

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

def Print_ngrams(doc):

    print("Token bigram:" , NGrams(tokens, 2))
    print("Token trigram:", NGrams(tokens, 3))

    pos = POS(doc)

    print("POS bigram:"   , NGrams(pos, 2))
    print("POS trigram:"  , NGrams(pos, 3))

#Print_ngrams()

"""
4. Lemmatization

"""   
def analysis_lemmatization():
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
#analysis_lemmatization()

"""
5. Named Entity Recognition

"""  
def analysis_NER():  
    #nlp(' '.join([str(item) for item in list(doc.sents)[:5]]))
  
    count = 0  
    ents = []
    labels = []
    for sent in doc.sents:        
        if(count == 5):
            break
        else:
            sentence = nlp(' '.join([str(item) for item in list(sent)]))
            for ent in sentence.ents:
                ents.append(ent.text)
                labels.append(ent.label_)
                print( (ent.text, ent.label_), ":")
            print(sentence)
            count += 1
    print("number of entities:", len(ents))
    print("number of unique labels:", len(set(labels)))

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

"""
7. Extract basic statistics

"""  

def basic_stat(data):
    
    prob = data.iloc[:,-1]
    binary = data.iloc[:,-2]
    target = data.iloc[:,-7]
 
    #Number of instances labeled with 0:
    count_0 = binary.value_counts()[0]
    print(count_0) 
    #Number of instances labeled with 1: 
    count_0 = binary.value_counts()[1]
    print(count_0)

    #Min, max, median, mean, and stdev of the probabilistic label: 
    min_prob = prob.min()
    max_prob = prob.max()
    median_prob = prob.median()
    mean_prob = prob.mean()
    sd_prob = prob.std()

    print(f"Min: {min_prob}\Max: {max_prob}\Median: {median_prob}\Mean: {mean_prob}\Stdv: {sd_prob}")

    #Number of instances consisting of more than one token: 
    #tokenize column
    df = pd.DataFrame()
    df["target"] = target
    df["count"] = target.apply(lambda x: len(word_tokenize(x)))

    # do summation on a column
    filter = df["count"] > 1
    print(filter.sum())
        
    #Maximum number of tokens for an instance:
    print(df["count"].max())
    #Estonia, Latvia, Lithuania, Romania, Bulgaria

"""
8. Linguistic characteristics

"""  
def analyze_ling(data):
    copy_df = data
    # filer on coplex and 1 token
    filter1 = copy_df.iloc[:,-2] == 1
    filter2 = copy_df.iloc[:, -7].apply(lambda x: len(word_tokenize(x))) == 1
    
    # filtering data on basis of both filters
    copy_df = copy_df[filter1 & filter2]

    # filter on index
    data = data.loc[copy_df.index]
   

    language = "en"
    complex = data.iloc[:,-7].tolist()
    complexity = data.iloc[:,-1].tolist()
    frequency = []
    length = []

    for i in complex:
        frequency.append(zipf_frequency(i,language))
        length.append(len(i))
        
    # peorson
    print(stats.pearsonr(length, complexity))
    print(stats.pearsonr(frequency, complexity))

    plot_scatter(length, complexity, "length", "complexity")
    plot_scatter(frequency, complexity, "frequency", "complexity")

def plot_scatter(x, y, xname, yname):
    fig, ax = plt.subplots()
    xticks = np.arange(0, (max(x)+1))
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_xticks(xticks)
    ax.set_title(f"Correlation of {xname} and {yname}")
    ax.scatter(x, y)
    plt.savefig(f"output/{xname}_{yname}.png")

if __name__ == "__main__":
    # This loads a small English model trained on web data.
    # For other models and languages check: https://spacy.io/models
    nlp = spacy.load('en_core_web_sm')
    # nltk.download('punkt')

    #file_name = r"C:\Users\janus\OneDrive\Msc Artificial Intelligence\Jaar 1\P5\NLP\assignments\intro2nlp_assignment1_code\intro2nlp_assignment1_code\data\preprocessed\train\sentences.txt"
    file = f"data/preprocessed/train/sentences.txt"
    with open(file, "r", encoding="utf8") as in_file:
        data = in_file.read().rstrip().replace('\n',' ')

    # Let's run the NLP pipeline on our test input
    doc = nlp(data)

    # # 1
    # tokenization(doc)
    #2
    # word_classes(doc)
    #3
    # tokens = list(filter(('\\"').__ne__, data.split())) # Take the raw data and filter out \\
    # Print_ngrams(doc)
    #4
    # analysis_lemmatization()
    #5
    # analysis_NER()
    #6
    """
    a. It marks the tart and end of (a) target word(s) in the sentence.
    b. It means that 40% of the annotators marked the word as complex.
    c. 8th and 9th columns show number of native vs non-native annotators who marked the target as difficult. 
        The binary is 1 after only 1 mark of difficult. The probability is native + non-native combined
     """ 
    #7
    file2 = f"data/original/english/WikiNews_Train.tsv"
    data2 = pd.read_csv(file2, sep='\t', header=None)

    # basic_stat(data2)
    #8
    analyze_ling(data2)