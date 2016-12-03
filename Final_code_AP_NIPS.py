
# coding: utf-8


#### LDA for AP dataset #####

## Author: Mridula Maddukuri 

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
from copy import deepcopy
import numpy
import matplotlib.pyplot as plt

# define stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "''","'s","``",u"n't",u'said']) # remove it if you need punctuation

def stemEachDoc(text):
    words = map(lambda word: word.lower(), word_tokenize(text))
    new_text = " ".join(map(lambda token: PorterStemmer().stem(token),words))
    return new_text

def stemAllDocs(docs):
    new_docs = deepcopy(docs)
    for i in range(len(docs)):
        new_docs[i] = stemEachDoc(docs[i])
    return new_docs
        
# explanation:
# text = "lover love lovely lovingly 512?"
# words = map(lambda word: word.lower(), word_tokenize(text))
# " ".join(map(lambda token: PorterStemmer().stem(token),words))
    

# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text)) # splitting string into words
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
    for i in range(len(docs)):
        new_words = tokenize(docs[i])
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

# vocab = extractVocabulary(docs)
# len(vocab) #26374
#new_docs = stemAllDocs(docs)
vocab = extractVocabulary_stemmed(docs)
len(vocab) #26145 # 37173 without stemming


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
#         for j in range(n):
#             N[j] = (np.transpose(R[:,j])*R[:,j])[0,0]
        N = numpy.linalg.norm(R, axis=0)
        jmax = numpy.argmax(N)
        J = J + [jmax]
        S = numpy.matrix(numpy.eye(m)) - R[:,jmax]*numpy.transpose(R[:,jmax])/(N[jmax])
        R = S*R
    Wt = M[:,J]    
    print 'Solving for A'
    At = numpy.linalg.lstsq(Wt,M)[0]
    return numpy.transpose(At),numpy.transpose(Wt)

# FUNCTION to visualize topic distributions in documents
# https://de.dariah.eu/tatom/topic_model_visualization.html
import random
def Visualize_doctopic(doc_topic,indices,filename):
    # stacked bar chart 
    a,b = doc_topic.shape
    width = 0.5
    #indices = random.sample(numpy.arange(a), 3)
    plots = []
    height_cumulative = numpy.zeros(len(indices))

    for i in range(b):
        color = plt.cm.coolwarm_r(float(i)/b,1)
        print color
        if i == 0:
            p = plt.bar([0,2,4],doc_topic[indices][:,i],width,color = color)
        else:
            p = plt.bar([0,2,4],doc_topic[indices][:,i],width,bottom = height_cumulative, color = color)
        height_cumulative = height_cumulative + doc_topic[indices][:,i]
        plots.append(p)

    plt.ylim((0,1))
    plt.xlim((0,8))
    plt.ylabel('Topics')
    plt.xlabel('Documents')
    plt.title('Topic distribution in randomly selected documents')
    plt.xticks(numpy.array([0,2,4]) + width/2, indices)
    plt.yticks(numpy.arange(0,1,10))
    topic_labels = ['Topic #{}'.format(k) for k in range(b)]
    plt.legend([p[0] for p in plots], topic_labels)
    plt.savefig(filename)
    plt.show()
    plt.close()
    return 


##### Using  LDA library : making X 
cv = CountVectorizer(vocabulary = vocab)
X = cv.fit_transform(docs).toarray()


### LDA library learning and results
import lda
#print numpy.where(~X.all(axis=0))[0]
model = lda.LDA(n_topics=5, n_iter=500, random_state=1)
model.fit(X) # see what n_iter is

topic_word = model.topic_word_
print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

doc_topic = model.doc_topic_
n = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    

#### LDA sklearn
from sklearn.decomposition import LatentDirichletAllocation
lda_sklearn = LatentDirichletAllocation(n_topics=5,random_state = 1,max_iter= 10)
lda_sklearn.fit(X)

lda_sklearn.components_.shape
n = 10
for i, topic_dist in enumerate(lda_sklearn.components_):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    
    

### Preprocessing for NMF : making X matrix

tfidf = TfidfVectorizer(tokenizer=tokenize_stemmed,
                        use_idf=True, sublinear_tf=False, max_features = 10000,
                        norm='l1');
##from docs
TF = tfidf.fit(docs)
TF_mat = tfidf.fit_transform(docs)
TF_mat.shape


#### NIPS ####### 
import numpy
with open('./TopicModelling/nips/vocab.nips.txt') as f:
    vocab_nips = f.read().split('\n')

A,B,C = numpy.loadtxt('./TopicModelling/nips/docword.nips.txt', skiprows = 3, unpack = True, dtype = int)
# buinding the doc-word matrix
R = numpy.zeros([1500, 12419])
R[A-1,B-1] = C 

R = TF_mat.todense()
print R.shape    
R = R[numpy.apply_along_axis(numpy.count_nonzero, 1, R) >= 20,:]
print R.shape

import lda
#print numpy.where(~X.all(axis=0))[0]
model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
model.fit(R.astype(int)) # see what n_iter is
doc_topic = model.doc_topic_
n = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab_nips)[numpy.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    
m,n = W.shape
print m,n
K = 20
for i in range(m):
    I = np.argsort(W[i,:]).tolist()[0][-K:]
    #print np.sort(W[i,:]).tolist()[0][-K:]
    print "Topic " + str(i)
    J = [vocab_nips[k] for k in I if W[i,k] != 0]
    print J 
    print ''


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm = 'l1')

TF_mat = transformer.fit_transform(R) 

#### Recursive NMF
# Y = TF_mat.todense()
from copy import deepcopy
A,W = fast_recursive_nmf(R,5)
m,n = W.shape
print m,n
K = 30
for i in range(m):
    I = numpy.argsort(W[i,:]).tolist()[0][-K:]
    #print np.sort(W[i,:]).tolist()[0][-K:]
    print "Topic " + str(i)
    J = [vocab[k] for k in I if W[i,k] != 0]
    print J 
    print ''
    
A_red = numpy.array(A)
row_sums = A_red.sum(axis=1)
A_new = A_red/ row_sums[:, numpy.newaxis]
a,b = A_new.shape
print a,b
A_red[1,:]



#### Library NMF
from sklearn.decomposition import NMF
nmf = NMF(n_components=5, random_state=1).fit(R)
W_lib = nmf.components_
A_lib = numpy.linalg.lstsq(numpy.matrix(W_lib).T,numpy.matrix(R).T)[0].T

m,n = W_lib.shape
print m,n
K = 20
for i in range(m):
    I = numpy.argsort(W_lib[i,:]).tolist()[-K:]
    #print np.sort(W[i,:]).tolist()[0][-K:]
    print "Topic " + str(i)
    J = [vocab[k] for k in I if W_lib[i,k] != 0]
    print J 
    print ''



row_sums = numpy.array(A_lib).sum(axis=1)
A_nlib = numpy.array(A_lib)/ row_sums[:, numpy.newaxis]
a,b = A_nlib.shape
print a,b

### to visualize 

Visualize_doctopic(A_nlib,indices,'AP_lib.png')




Visualize_doctopic(doc_topic,indices,'AP.png') # enter a list of THREE indices




# Example to illustrate CV and TFIDF

trial = ["apple banana mango","apple mango apple",'apple apple banana apple']
vocab = ['apple','mango','banana']
Cv_t = CountVectorizer()
X1 =Cv_t.fit_transform(trial).toarray()
from sklearn.feature_extraction.text import TfidfTransformer
Tfifd = TfidfTransformer(smooth_idf=True, norm = 'l1')

X2 = Tfifd.fit_transform(X1).toarray() 
print X1
print X2



# #NYT sub-datset preprocessing

# matrix=[]
# with open('./TopicModelling/NYT/docword.nytimes.txt') as f:
#     for i in range(3):
#         f.next()
#     for line in f:
#         matrix.append(line)
#         i = i + 1
#         if i > 4000000:
#             break

            
# A=[0]*len(matrix)
# B=[0]*len(matrix)
# C=[0]*len(matrix)
# for i in range(len(matrix)):
#     full_array = numpy.fromstring(matrix[i], dtype=int, sep=" ")
#     A[i]=full_array[0]
#     B[i]=full_array[1]
#     C[i]=full_array[2]

# numpy.array(B)
# #creating matrix and reducing number of words
# with open('./TopicModelling/NYT/vocab.nytimes.txt') as f:
#     vocab_NYT = f.read().split('\n')
# m = numpy.max(A)
# n = numpy.max(B)
# I = numpy.random.choice(n,15000)
# I_set = set(I)
# vocab_NYT_reduced = [vocab_NYT[i] for i in I]
# n_reduced = 15000
# R = numpy.zeros([m,n_reduced])
# dic = {}
# for i in range(len(I)):
#     dic[I[i]] = i
# for i in range(len(matrix)):
#     if B[i] in I_set:
#         R[A[i]-1,dic[B[i]]-1] = C[i]
# R_NYT = deepcopy(R)




# #### NMF #####
# The ground truth of NMF: all the documents are convex combinations of few pure topics
# # Idea given by ned: 
# -> why not learn first 500 documents 
# -> cluster 
# ---> use the labels to classify the other documents
 

