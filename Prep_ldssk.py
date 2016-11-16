import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer

import Preprocessing

LDA_instance = Preprocessing.PreprocessDocs()
LDA_instance.extractVocabulary()  

# def top_words(num_of_topics, count):
#     for i in range(num_of_topics):
#     print "Topic " +str(i) +": Top words"
#     idx = (-topic_vocab[i]).argsort()[:count]
#     Y = [LDA_instance.vocab[l] for l in list(idx)]
#     print Y


#### Using SKlearn library
number_of_topics = 3
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics= number_of_topics, max_iter = 5, learning_method='online', learning_offset=50., random_state=0)
lda.fit(LDA_instance.makeDocVocabMatrix())

# Document - topic matrix
lda.transform(LDA_instance.makeDocVocabMatrix())

# Topic - vocab
lda.components_







