import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import matplotlib
import pandas

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
        self.pos = lda.fit_transform(tf)
        self.xs, self.ys = self.pos[:, 0], self.pos[:,1]

    def plot_lda_results(self):
        ax = plt.subplot(111)
        for label,marker,color in zip(
            range(0,10),('^', 's', 'o','^', 'o', 'o','^', '^', 'o','s'),('blue', 'red', 'green','blue', 'red', 'green','blue', 'red', 'green', 'yellow')):
            plt.scatter(x=self.xs , y=self.ys, marker=marker, color=color, alpha=0.5,label=label)
        plt.xlabel('LD1')
        plt.ylabel('LD2')

        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('LDA: Result')
        # hide axis ticks
        plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)    
        plt.grid()
        plt.tight_layout
        plt.show()

        
if __name__ == "__main__":
    new = PreprocessDocs()
    new.extractVocabulary()
    #print new.vocab
    #print new.doc_list
    #print new.makeDocVocabMatrix()
    n = LDASklearn()
    n.ldafit()
    n.plot_lda_results()
    
    
'''#set up cluster names using a dict
cluster_names = {0: 'Family, home, war', 
                 1: 'Police, killed, murders', 
                 2: 'Father, New York, brothers', 
                 3: 'Dance, singing, love', 
                 4: 'Killed, soldiers, captain'}

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)'''

'''        #print "\nTopics in LDA model:"
        #tf_feature_names = tf_vectorizer.get_feature_names()
        #self.print_top_words(lda, tf_feature_names, self.n_top_words)
        xs, ys = pos[:, 0], pos[:,1]
        #set up colors per clusters using a dict
        cluster_colors = {0: '#lb9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
        #Cluster names can be added as dictionary
        df = pandas.DataFrame(dict(x=xs, y=ys,))
        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) 
        ax.plot(df.x, df.y, marker='o',linestyle= '',ms =12)
        plt.show()'''
    