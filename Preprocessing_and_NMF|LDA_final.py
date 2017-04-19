
# coding: utf-8


#### Preprocessing #####

## Author: Mridula Maddukuri 

# # to check how fast the code snippet is 
from timeit import default_timer as timer
# start = timer()
# end = timer()    
# print(end-start)
import glob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
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
from sklearn.decomposition import NMF
import random

class PreprocessDocs_NMF:
    def __init__(self, filename = './TopicModelling/ap/ap.txt'):
        """Input is essentially a folder of txt files. change the path in airquotes as need be"""
        #list_of_files = glob.glob('*.txt')
        #doc_list = []
        with open(filename) as f:
            docs = f.read().split('\n')
        # for File in list_of_files:
        #     with open(File) as f:
        #         doc = f.read()
        #         doc_list = doc_list + [doc]
        doc_list = [w for w in docs if len(w) > 50]
        self.doc_list = doc_list
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "''","'s","``",u"n't",u'said']) # remove it if you need punctuation



    def stemEachDoc(self,text):
        words = map(lambda word: word.lower(), word_tokenize(text))
        new_text = " ".join(map(lambda token: PorterStemmer().stem(token),words))
        return new_text

    def stemAllDocs(self,docs):
        new_docs = deepcopy(self.doclist)
        for i in range(len(docs)):
            new_docs[i] = stemEachDoc(docs[i])
        return new_docs

    # https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/       
    def tokenize(self,text):
        """ Stemming included"""
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text)) # splitting string into words
        words = [word for word in words if word not in self.stop_words]
        tokens =(list(map(lambda token: PorterStemmer().stem(token),words))) # extracting the stems 
        p = re.compile('[a-zA-Z]+') # to remove numbers from text
        filtered_tokens = list(filter(lambda token:p.match(token) and len(token)>=min_length,tokens));
        return filtered_tokens

    def tokenize_stemmed(self,text):
        """ Stemming not included"""
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in self.stop_words]
        p = re.compile('[a-zA-Z]+')
        filtered_words = list(filter(lambda token:p.match(token) and len(token)>=min_length,words))
        return filtered_words

    def extractVocabulary(self):
        # """Returns vocabulary from a list of stemmed documents"""
        word_list = []
        for i in range(len(self.doc_list)):
            new_words = self.tokenize(self.doc_list[i])
            word_list = word_list + new_words
            #print len(word_list)
        self.vocab = list(set(word_list))
        return self.vocab

    def extractVocabulary_stemmed(self):
        # """Returns vocabulary from a list of unstemmed documents"""
        word_list = []
        for i in range(len(self.doc_list)):
            new_words = self.tokenize_stemmed(self.doc_list[i])
            word_list = word_list + new_words
            #print len(word_list)
        self.vocab = list(set(word_list))
        return self.vocab 

    
    def makeDocVocabMatrix(self, WhichAlgorithm):
        """Given a list of documents, returns a matrix with rows(each document) consisting of word frequency"""
        if WhichAlgorithm == "LDA":
            cv = CountVectorizer(vocabulary= self.vocab)
            return cv.fit_transform(self.doc_list).toarray()
        elif WhichAlgorithm == "NMF":
            tfidf = TfidfVectorizer(tokenizer=self.tokenize_stemmed,
                        use_idf=True, sublinear_tf=False, max_features = 10000,
                        norm='l1');
            TF = tfidf.fit(new.doc_list)
            TF_mat = tfidf.fit_transform(self.doc_list)
            return TF_mat.toarray()
    
    def Visualize_doctopic(self,doc_topic,indices,filename):
        """ indices:  a list of 3 random indices
            filename: the file you want to save the plot to
            doc_topic matrix you get from  WhichAlgorithm you used"""
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

    
    def fast_recursive_nmf(self,Y,r):
        """ Given the doc-vocab matrix and number of topics, returns the factorized matrices"""
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
    
    
        
if __name__ == "__main__":
    new = PreprocessDocs()
    new.extractVocabulary_stemmed()
    print len(new.vocab)
    print len(new.doc_list)

    X = new.makeDocVocabMatrix("NMF")

    # FAST-Recursive NMF
    A,B = new.fast_recursive_nmf(X,5)
    K = 20
    m = 5
    for i in range(m):
        I = numpy.argsort(B[i,:]).tolist()[-K:]
        #print np.sort(W[i,:]).tolist()[0][-K:]
        print "Topic " + str(i) +" Top words"
        J = [new.vocab[k] for k in I[0] if B[i,k] != 0]
        print J 
        print ''

    indices = [1,10,20]
    new.Visualize_doctopic(A,indices,'NMF.png')

    """ Library NMF"""
    # ### Library NMF
    # nmf = NMF(n_components=5, random_state=1).fit(X)
    # W_lib = nmf.components_
    # A_lib = numpy.linalg.lstsq(numpy.matrix(W_lib).T,numpy.matrix(R).T)[0].T

    # m,n = W_lib.shape
    # print m,n
    # K = 20
    # for i in range(m):
    #     I = numpy.argsort(W_lib[i,:]).tolist()[-K:]
    #     #print np.sort(W[i,:]).tolist()[0][-K:]
    #     print "Topic " + str(i)
    #     J = [vocab[k] for k in I if W_lib[i,k] != 0]
    #     print J 
    #     print ''

    """ To execute LDA using the lda library"""
    #X_lda = new.makeDocVocabMatrix("LDA")    
    # import lda
    # #print numpy.where(~X.all(axis=0))[0]
    # model = lda.LDA(n_topics=10, n_iter=500, random_state=1)
    # model.fit(R.astype(int)) # see what n_iter is
    # doc_topic = model.doc_topic_
    # n = 10
    # topic_word = model.topic_word_
    # for i, topic_dist in enumerate(topic_word):
    #     topic_words = numpy.array(vocab_nips)[numpy.argsort(topic_dist)][:-(n+1):-1]
    #     print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
        
    # m,n = W.shape
    # print m,n
    # K = 20
    # for i in range(m):
    #     I = np.argsort(W[i,:]).tolist()[0][-K:]
    #     #print np.sort(W[i,:]).tolist()[0][-K:]
    #     print "Topic " + str(i)
    #     J = [vocab_nips[k] for k in I if W[i,k] != 0]
    #     print J 
    #     print ''

    


    
            