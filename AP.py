#### LDA for AP dataset #####

## Author: Mridula Maddukuri 

"""references: 
https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       

"""

# # to check how fast the code snippet is 
from timeit import default_timer as timer
# start = timer()
# end = timer()    
# print(end-start)



#### READ DATA ####
with open('/Users/mridulamaddukuri/Dropbox/python/Computational Optimization/Project/TopicModelling/ap/ap.txt') as f:
    docs = f.read().split('\n')
# get rid of any expression lwith length ess than 50. This gets rid of <TEXT> like expressions.
docs = [w for w in docs if len(w) > 50]


#### DOCUMENT REPRESENTATION ####
from nltk.corpus import stopwords # stop words
from nltk.tokenize import wordpunct_tokenize,word_tokenize # splits sentences into words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem.lancaster import LancasterStemmer # extract the roots of words 
from nltk.stem.porter import PorterStemmer # extract the roots of words 
import re
import lda
from copy import deepcopy
import numpy
import matplotlib.pyplot as plt 


# define stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation

# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       

def stemEachDoc(text):
    words = map(lambda word: word.lower(), word_tokenize(text))
    new_text = " ".join(map(lambda token: PorterStemmer().stem(token),words))
    return new_text

def stemAllDocs(docs):
    new_docs = deepcopy(docs)
    for i in range(len(docs)):
        new_docs[i] = stemEachDoc(docs[i])
    return new_docs

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text)); # splitting string into words
    words = [word for word in words if word not in stop_words]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),words))) # extracting the stems 
    p = re.compile('[a-zA-Z]+') # to remove numbers from text
    filtered_tokens = list(filter(lambda token:p.match(token) and len(token)>=min_length,tokens));
    return filtered_tokens

def tokenize_stemmed(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stop_words]
    p = re.compile('[a-zA-Z]+')
    filtered_words = list(filter(lambda token:p.match(token) and len(token)>=min_length,words))
    return filtered_words

def extractVocabulary(docs):
    # """Returns vocabulary from a list of documents"""
    word_list = []
    for doc in docs:
        new_words = tokenize(doc)
        word_list = word_list + new_words
        #print len(word_list)
    vocab = list(set(word_list))
    return vocab

def extractVocabulary_stemmed(docs):
    # """Returns vocabulary from a list of documents"""
    word_list = []
    for i in range(len(docs)):
        new_words = tokenize_stemmed(docs[i])
        word_list = word_list + new_words
        #print len(word_list)
    vocab = list(set(word_list))
    return vocab

#vocab = extractVocabulary(docs)
# len(vocab) # 26374

# new_docs = stemAllDocs(docs)
# the above step is to convert all words in each document to their respective stems: to remove redundancy

# new_vocab = extractVocabulary_stemmed(new_docs)
# #len(new_vocab) #26348

# excluding stemming because it's not giving words that make sense
vocab = extractVocabulary_stemmed(docs)
len(vocab) #26145 # 37173 without stemming


####### LDA LIBRARY ########
# for lda library : countVectorizer is better 
# counting the occurrences of tokens in each document
# for LDA using lda library
cv = CountVectorizer(vocabulary= vocab)
lda_X = cv.fit_transform(docs).toarray()

model = lda.LDA(n_topics=5, n_iter=500, random_state=1)
model.fit(lda_X)

topic_word = model.topic_word_
n = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

model.doc_topic_ # gives the probability distribution over topics for each document


###### LDA - sklearn #####
from sklearn.decomposition import LatentDirichletAllocation
model2 = LatentDirichletAllocation(n_topics = 5, random_state = 1,max_iter = 10)
model2.fit(lda_X)

# top 10 words in each topic
n = 10
for i, topic_dist in enumerate(lda_sklearn.components_):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))




########## NMF ############
"""Based on the paper : https://arxiv.org/pdf/1208.1237v3.pdf""" 
# r is the number of topics
def fast_recursive_nmf(Y,r):
    M = numpy.transpose(numpy.matrix(Y.astype(float)))
    J = []
    m,n = M.shape
    R = deepcopy(M)
    # print Y.shape
    # print R.shape
    # print R
    for i in range(n):
        R[:,i] = R[:,i]/numpy.sum(R[:,i])
    print 'normalized'
    for i in range(r):
        print 'topic ' + str(i)
        N = numpy.array([0.0]*n)
        for j in range(n):
            N[j] = (numpy.transpose(R[:,j])*R[:,j])[0,0]
        jmax = numpy.argmax(N)
        J = J + [jmax]
        S = numpy.matrix(numpy.eye(m)) - R[:,jmax]*numpy.transpose(R[:,jmax])/(N[jmax])
        R = S*R
    Wt = M[:,J] # topic vocab 
    print 'Solving for A'
    At = numpy.linalg.lstsq(Wt,M)[0] # document topic
    return numpy.transpose(At),numpy.transpose(Wt)

### vectorize using tfidf #####
tfidf = TfidfVectorizer(tokenizer=tokenize_stemmed,
                        use_idf=True, sublinear_tf=False, max_features = 5000,
                        norm='l1');
"""L1 norm is very important here. in the second for loop of fast_recursive_nmf function, the document with 
highest L2 norm is picked. if you passed a tfidf with L2 norm, them all of the documents will have an L2 norm of 1.
Then the 'pure documents' are picked at random since there are multiple documents with L2 norm = 1 """

TF = tfidf.fit(docs)
TF_mat = tfidf.fit_transform(docs)
# TF_mat is a sparse matrix ~ obviously
nmf_X = TF_mat.todense()

Tf_vocab = TF.vocabulary_ 
rev_TF_vocab = {}
for k in Tf_vocab:
	rev_TF_vocab[Tf_vocab[k]] = k

doc_top,topic_vocab = fast_recursive_nmf(nmf_X,5)
m,n = topic_vocab.shape
K = 30
for i in range(m):
    I = np.argsort(W[i,:]).tolist()[0][-K:]
    #print np.sort(W[i,:]).tolist()[0][-K:]
    print "Topic " + str(i)
    J = [revTF_vocab[k] for k in I if W[i,k] != 0]
    print J 
    print ''


# for NMF : tfidf is better because it gives less importance to trivial words like 'the'
# TfidfVectorizer: tokenizing strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
# TfidfVectorizer itself tokenizes the documents! :|
# WHAT ARE FEATURES AND SAMPLES?
# In this scheme, features and samples are defined as follows:
# each individual token occurrence frequency (normalized or not) is treated as a feature.
# the vector of all the token frequencies for a given document is considered a multivariate sample.


##### VECTORIZATION ##### general process of turning a collection of text documents into numerical feature vectors







