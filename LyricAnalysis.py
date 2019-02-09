import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os
import glob
import errno
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('corpus')

sys.setdefaultencoding('utf-8')

###################################
# Importing data
###################################

path = '/allLyrics/*.txt' #note C:
files = glob.glob(path)
lyric_data = []
titles = []

for name in files:
    titles.append(os.path.splitext(name)[0])
    try:
        with open(name) as f:
            lyric_data.extend(f)
    except IOError as exc: #Not sure what error this is
        if exc.errno != errno.EISDIR:
            raise


###################################
# Tokenizing and Stemming
###################################

# Use TinySegmenter for Japanese tokenizing and stemming
import tinysegmenter
import codecs
f = codecs.open('ja_stopword.txt', 'r', 'utf-8')
jpn_stopwords = set([x[:-1] for x in f.readlines()]) # strip \n
f.close()

def jpn_parse(sent):
    stem =  tinysegmenter.TinySegmenter() 
    stop = jpn_stopwords
    tx = stem.tokenize(sent)
    px = [x for x in tx if x not in stop]
    return px

docs_stemmed = []

for i in lyric_data:
    tokenized_and_stemmed_results = jpn_parse(i)
    docs_stemmed.extend(tokenized_and_stemmed_results)

##########################################################
# Text Vectorization
##########################################################

# Using TF-IDF model for vectorization
tfidf_model = TfidfVectorizer(max_df=0.8, max_features=2000,
                                 min_df=0, stop_words=jpn_stopwords,
                                 use_idf=True, tokenizer=jpn_parse, ngram_range=(1,3))

tfidf_matrix = tfidf_model.fit_transform(lyric_data) #fit the vectorizer to lyric data

print("In total, there are " + str(tfidf_matrix.shape[0]) + \
      " documents and " + str(tfidf_matrix.shape[1]) + " terms.")

tf_selected_words = tfidf_model.get_feature_names()

# Use cosine similarity to check the similarity between each pair of documents
from sklearn.metrics.pairwise import cosine_similarity
cos_matrix = cosine_similarity(tfidf_matrix)
print(cos_matrix)


################################################################
# Clustering Analysis
###############################################################

# Use k-means clustering
from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# Create a new DataFrame from all of the input files.
songs = { 'title': titles,  'lyrics': lyric_data, 'cluster': clusters}
frame = pd.DataFrame(songs, index = [clusters] , columns = ['title', 'cluster'])
frame.head(10)

print("Number of songs included in each cluster:")
frame['cluster'].value_counts().to_frame()

print("<Document clustering result by K-means>")

#km.cluster_centers_ denotes the importances of each items in centroid.

# Sort it in decreasing-order.
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

Cluster_keywords_summary = {}
for i in range(num_clusters):
    print("Cluster " + str(i) + " words:", end='')
    Cluster_keywords_summary[i] = []
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        Cluster_keywords_summary[i].append(vocab_frame_dict[tf_selected_words[ind]])
        print(vocab_frame_dict[tf_selected_words[ind]] + ",", end='')
    print()
    # Here ix means index, which is the clusterID of each item.
    # Without tolist, the values result from dataframe is <type 'numpy.ndarray'>
    cluster_songs = frame.ix[i]['title'].values.tolist()
    print("Cluster " + str(i) + " titles (" + str(len(cluster_songs)) + " songs): ")
    print(", ".join(cluster_songs))
    print()

# Use PCA to reduce dimensions to 2d and try for a visualization

pca = decomposition.PCA(n_components=2)
tfidf_matrix_np=tfidf_matrix.toarray()
pca.fit(tfidf_matrix_np)
X = pca.transform(tfidf_matrix_np)

xs, ys = X[:, 0], X[:, 1]

#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
#set up cluster names using a dict
cluster_names = {}
for i in range(num_clusters):
    cluster_names[i] = ", ".join(Cluster_keywords_summary[i])

# %matplotlib inline 
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 
groups = df.groupby(clusters)

fig, ax = plt.subplots(figsize=(16, 9))

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')

ax.legend(numpoints=1,loc=4)  

plt.show() 

plt.close()

##################################################
# Topic Analysis
##################################################
# Use LDA for clustering
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=5, learning_method = 'online')

tfidf_matrix_lda = (tfidf_matrix * 100)
tfidf_matrix_lda = tfidf_matrix_lda.astype(int)
lda.fit(tfidf_matrix_lda)

topic_word = lda.components_
print(topic_word.shape)

n_top_words = 7
topic_keywords_list = []
for i, topic_dist in enumerate(topic_word):
    # select top(n_top_words-1)
    lda_topic_words = np.array(tf_selected_words)[np.argsort(topic_dist)][:-n_top_words:-1] 
    for j in range(len(lda_topic_words)):
        lda_topic_words[j] = vocab_frame_dict[lda_topic_words[j]]
    topic_keywords_list.append(lda_topic_words.tolist())

# Print out the clusters and topics and titles of the songs
topic_doc_dict = {}
print("<Document clustering result by LDA>")
for i in range(len(doc_topic)):
    topicID = doc_topic[i].argmax()
    if topicID not in topic_doc_dict:
        topic_doc_dict[topicID] = [titles[i]]
    else:
        topic_doc_dict[topicID].append(titles[i])
for i in topic_doc_dict:
    print("Cluster " + str(i) + " words: " + ", ".join(topic_keywords_list[i]))
    print("Cluster " + str(i) + " titles (" + str(len(topic_doc_dict[i])) + " songs): ")
    print(', '.join(topic_doc_dict[i]))
    print()


