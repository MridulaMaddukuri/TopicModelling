import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.decomposition import LatentDirichletAllocation

class PreprocessDocs:
    def __init__(self):
        """Input is essentially a folder of txt files. change the path in airquotes as need be"""
        list_of_files = glob.glob('*.txt')
        doc_list = []
        for File in list_of_files:
            with open(File) as f:
                doc = f.read()
                doc_list = doc_list + [doc]
        self.doc_list = doc_list
        
    def extractVocabulary(self):
        """Returns vocabulary from a list of documents"""
        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation
        st = LancasterStemmer()
        word_list = []
        for doc in self.doc_list:
            new_words = [i.lower() for i in wordpunct_tokenize(doc) if i.lower() not in stop_words]
            # If you want to remove redundancy in terms of pularity etc
            #new_words2 = [(st.stem(i)).lower() for i in wordpunct_tokenize(doc) if i.lower() not in stop_words]
            word_list = word_list + new_words
        #print len(word_list)
        self.vocab= list(set(word_list))
        # print len(vocab)
        # return vocab
    
    def makeDocVocabMatrix(self):
        """Given a list of documents, returns a matrix with rows(each document) consisting of word frequency"""
        cv = CountVectorizer(vocabulary= self.vocab)
        return cv.fit_transform(self.doc_list).toarray()
    

class LDASklearn:
    def __init__(self):
        self.n_samples = 2000
        self.n_features = 1000
        self.n_topics = 10
        self.n_top_words = 20
        #self.tf = tf
        """Input is essentially a folder of txt files. change the path in airquotes as need be"""
        list_of_files = glob.glob('*.txt')
        doc_list = []
        for File in list_of_files:
            with open(File) as f:
                doc = f.read()
                doc_list = doc_list + [doc]
        self.doc_list = doc_list
        self.data_samples = self.doc_list[:self.n_samples]

    def print_top_words(self,model, features_names, n_top_words):
        for idx, topic in enumerate(model.components_):
            print "Topics No%d:"%idx
            print " ".join([features_names[i] for i in topic.argsort()[:-n_top_words-1:-1]])
    
    def ldafit(self):
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df = 2, max_features = self.n_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(self.data_samples)
        print "Fitting LDA models with tf features"
        lda = LatentDirichletAllocation(n_topics=self.n_topics, max_iter = 5, learning_method='online', learning_offset=50., random_state=0)
        lda.fit(tf)
        print "\nTopics in LDA model:"
        tf_feature_names = tf_vectorizer.get_feature_names()
        self.print_top_words(lda, tf_feature_names, self.n_top_words)
    
        
if __name__ == "__main__":
    new = PreprocessDocs()
    new.extractVocabulary()
    #print new.vocab
    #print new.doc_list
    #print new.makeDocVocabMatrix()
    n = LDASklearn()
    n.ldafit()
    
    
    