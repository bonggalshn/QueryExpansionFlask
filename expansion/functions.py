import PyPDF2
import os
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ast

import math
from collections import OrderedDict
from operator import itemgetter
# ------------------------------------------------------------------------
from flask import url_for, session
from expansion import app

def allFile(location):
    document = []
    for doc in os.walk(location):
        document = doc[2]
    return document

# return list of document's content (input: string path)
def extractPDF(location):
    documents = allFile(location)
    allText = []
    for doc in documents:
        file = open(location+'/'+doc, 'rb')
        fileReader = PyPDF2.PdfFileReader(file)
        
        docs = ''
        pages = fileReader.numPages
        for page in range(pages):
            obj = fileReader.getPage(page)
            docs = docs + obj.extractText()
        allText.append(docs)
    return allText

# return list of number as documentID (input: string path)
def generateDocNumber(filename):
    docNum = []
    for file in filename:
        docNum.append(str(filename.index(file)))
    return docNum

# PREPROCESSING
# return list of string (input: list of sentence)
def removePunctuation(textList):
    for i in range(len(textList)):
        for punct in string.punctuation:
            textList[i] = textList[i].replace(punct, " ")
        textList[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', textList[i], flags=re.MULTILINE)
    return textList

# return list of string (input: list of sentence)
def caseFolding(textList):
    text = []
    for i in range(len(textList)):
        text.append(textList[i].lower())
    return text

# return list of word (input: sentence)
def token(sentence):
    token = []
    for word in CountVectorizer().build_tokenizer()(sentence):
        token.append(word)
    return token

# return list of word list (input: list of sentence)
def tokenize(textList):
    tokens = []
    for i in range(len(textList)):
        tokens.append(token(textList[i]))
    return tokens

# return string of sentence (input: sentence and list of stop words)
def checkStopword(sentence, stop_words):
    sentence = [w for w in sentence if not w in stop_words]
    return sentence
    
# return list of word list (input: list of word list)
def stopwordRemove(textList):
    stop_words = set(stopwords.words('english'))
    text = []
    for i in range(len(textList)):
        text.append(checkStopword(textList[i], stop_words))
    return text

# return list of word list (input: list of word list)
def numberRemove(textList):
    text = []
    for i in range(len(textList)):
        text.append([w for w in textList[i] if not any(j.isdigit() for j in w)])
    return text

# return list of word list (input: list of word list)
def stemming(textList):
    stemmer = PorterStemmer()
    text = textList
    for i in range(len(textList)):
        for j in range(len(textList[i])):
            text[i][j] = stemmer.stem(text[i][j])
    return text

# return list of word (input: list of word)
def sorting(textList):
    for i in range(len(textList)):
        textList[i] = sorted(textList[i])
    return textList

# return list of word list (input: list of word list)
def getAllTerms(textList):
    terms = []
    for i in range(len(textList)):
        for j in range(len(textList[i])):
            terms.append(textList[i][j])
    return sorted(set(terms))

# INDEXING FUNCTION
# return index as dictionary (input: list of word list and integer document number)
def createIndex(textList, docno):
    terms = getAllTerms(textList)
    proximity = {}
    for term in terms:
        position = {}
        for n in range(len(textList)):
            if(term in textList[n]):
                position[docno[n]] = []
                for i in range(len(textList[n])):
                    if(term == textList[n][i]):
                        position[docno[n]].append(i)
        proximity[str(term)] = position
    return proximity

# (input: dictionary of index and string of index's filename)
def exportIndex(index, filename):
    file = open(filename,'w')
    for n in index:
        file.write(str(n)+'\n')
        for o in index[n]:
            file.write('\t'+o+': ')
            for p in range(len(index[n][o])):
                file.write(str(index[n][o][p]))
                if(p<len(index[n][o])-1):
                    file.write(', ')
                else:
                    file.write('\n')
    file.close()
    return "Index's file has been successfully created."

def saveIndex(index):
    index = str(index)
    file = open("Index.txt", 'w')
    file.write(index)
    file.close()
    return 0


# RANKED RETRIEVAL FUNCTION
# return list of word found in index (input: list of query and dictionary of index)
def queryInIndex(query, index):
    result = []
    for word in query:
        if word in index:
            result.append(word)
    return result


def df(query, index):
    docFreq = {}
    for word in query:
        if word in index:
            docFreq[word] = len(index[word])
    return docFreq

def idf(df, N):
    inv = {}
    for word in df:
        inv[word] = math.log10(N/df[word])
    return inv

def tf(query, index):
    termFreq = {}
    for word in query:
        freq = {}
        if word in index:
            for i in index[word]:
                freq[i] = len(index[word][i])
        termFreq[word] = freq
    return termFreq

def tfidf(tf, idf):
    w = {}
    for word in tf:
        wtd = {}
        for doc in tf[word]:
            wtd[doc] = (1+(math.log10(tf[word][doc])))*idf[word]
        w[word] = wtd
    return w
    
def score(TFIDF):
    res = {}
    # ScoreResult = {}

    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = 0
    for i in TFIDF:
        for j in TFIDF[i]:
            res[j] = res[j] + TFIDF[i][j]
            
    # sorted_dict = sorted(res, key=res.get, reverse=True)

    resss = OrderedDict(sorted(res.items(), key=itemgetter(1), reverse=True))

    # print("/n/nResult===> ", list(resss.values()))
    return resss

def generateIndex():
	location = "../QueryExpansion/expansion/collection"
	filename = allFile(location) 
	extracted= extractPDF(location)
	# totalDoc = len(filename)
	documentNumber = generateDocNumber(filename)

	for i in range(len(filename)):
		extracted[i] = str(extracted[i].encode("utf-8"))

	# PREPROCESSING
	text = removePunctuation(extracted)
	text = caseFolding(text)
	text = tokenize(text)
	text = stopwordRemove(text)
	text = numberRemove(text)
	text = stemming(text)

	# GET ALL TERMS IN COLLECTION
	# terms = getAllTerms(text)

	# INDEXING

	# index = createIndex(text,documentNumber, terms)
	index = createIndex(text,documentNumber)

	# EXPORT INDEX FILE
	# exportIndex(index, 'INDEX_PDF.txt')
	return index

def all_filename():
    location = "../QueryExpansion/expansion/collection"
    filename = allFile(location)
    return filename

def all_content():
    location = "../QueryExpansion/expansion/collection"
    extracted= extractPDF(location)
    filename = allFile(location)
    for i in range(len(filename)):
        extracted[i] = str(extracted[i].encode("utf-8"))
    return extracted


def search(user_query, prox_index):
    index = prox_index

    # QUERY
    # String of query which will be searched
    raw_query = [user_query]

    location = "../QueryExpansion/expansion/collection"
    filename = allFile(location)
    totalDoc = len(filename)

    # preprocess the query
    query = removePunctuation(raw_query)
    query = caseFolding(query)
    query = tokenize(query)
    query = stopwordRemove(query)
    query = numberRemove(query)
    query = stemming(query)
    query = query[0]

    # check query in the index
    query = queryInIndex(query, index)


    # RANKED RETRIEVAL
    N               = totalDoc
    # tfidf_list      = []

    docFrequency    = df(query, index) # type: dictionary of 
    invDocFrequency = idf(docFrequency, N) # type: dictionary of double
    termFrequency   = tf(query, index) # type: dictionary of
    TFIDF           = tfidf(termFrequency, invDocFrequency) # type: dictionary of
    sc              = score(TFIDF)  # type: list of
    # print(list(sc.keys()))
    return sc

def preprocess(extracted): # list -> ['a b c', 'd e f']
    text = removePunctuation(extracted)
    text = caseFolding(text)
    text = tokenize(text)
    text = stopwordRemove(text)
    text = numberRemove(text)
    text = stemming(text)
    return text # list -> [['a','b','c'],['d','e','f']]

# ==============================================================================================================================
# EXPANSION FUNCTIONS
# ==============================================================================================================================

def relevance(listDoc, allContent):
    relevance = []
    # irrelevance = []
    # print("listDoc======>", listDoc)
    # print("allContent", allContent)
    if len(listDoc) < 5:
        maxLen = len(listDoc)
    else:
        maxLen = 5

    for i in range(maxLen):
        if (i < 5):
            relevance.append(allContent[int(listDoc[i])])
        # else:
            # irrelevance.append(allContent[int(listDoc[i])])

    # relevances = preprocess(relevance)

    # result = {"rel":relevances,"irrel":irrelevances}
    result = {"rel": relevance}

    return result

# def notRelevance(rel, allNumber, allContent):
#     notRelevance = []
#     notRelevanceNumber = []
#     rel = [int(x) for x in rel]

#     for i in allNumber:
#         if int(i) not in (rel):
#             notRelevanceNumber.append(int(i))
    
#     for i in notRelevanceNumber:
#         notRelevance.append(allContent[int(i)])
    
#     notRelevance = preprocess(notRelevance)
#     return notRelevance # list -> [['a','b'],['c','d']]


def vector(text, terms): # text -> ['a','b','c'] | terms -> ['b','c','d'] : all terms
    Vec = []
    for i in range(len(terms)):
        if (terms[i] in text):
            Vec.append(1)
        else:
            Vec.append(0)
    return Vec # list -> [ 1 , 1 , 0 ]

def sumVector(VectorList):  # [[[1,1,0],[1,0,1]], [[1,1,0],[1,0,1]]]
    results = []
    results.append(VectorList[0])
    # result[0][0]

    for i in range(1, len(VectorList)):
        for j in range(len(VectorList[i])):
            results[0][j] = results[0][j] + VectorList[i][j] 
    

    # results[0][0] = results[0][0] + 1

    # for i in range(1, len(VectorList)):
    #     for j in range(len(VectorList[i])):
    #         results[j] = results[j] + 1 

        # for i in range(1,len(VectorList)):
        #     for j in range(len(VectorList[i])):
        #         results[i][j] = results[i][j] + 1 
                # result[i][j] = int(int(result[i][j]) + int(VectorList[i][j]))
    return results[0]

def multiplyVector(alpha, listVector):  # alpa = float, listVector = [1,2,3]
    res = []
    for i in range(len(listVector)):
        res.append(alpha*int(listVector[i]))
    return res
# def preExpansion(query, terms, rel, irrel):
#     queryVec      = vector(query, terms)
#     relevanVec    = []
#     notRelevanVec = []

#     for i in rel:
#         relevanVec.append(vector(text[i], terms))
    
#     for i in irrel:
#         notRelevanVec.append(vector(text[i], terms))
#     return "result"

# def expansion(old_query, allTerms):
#     queryVec = vector(old_query,allTerms)


