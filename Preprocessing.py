import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer

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
    
    
    
        
if __name__ == "__main__":
    new = PreprocessDocs()
    new.extractVocabulary()
    print new.vocab
    print new.doc_list
    print new.makeDocVocabMatrix()
    
    
            