import streamlit as st
import pandas as pd
import numpy as np
import pickle

# import random
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import umap
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer


###############################
st.write("# Loading resources")
###############################


with open('POC_sentence_transformer.pkl', 'rb') as file:
    sentence_transformer = pickle.load(file)
st.write("sentence_transformer loaded")


with open('POC_umap_reducer.pkl', 'rb') as file:
    umap_reducer = pickle.load(file)
st.write("umap_reducer loaded")


with open('POC_hdbscan_clusters.pkl', 'rb') as file:
    hdbscan_clusters = pickle.load(file)
st.write("hdbscan_clusters loaded")

with open('aspect_extractor.pkl', 'rb') as file:
    aspect_extractor = pickle.load(file)
st.write("aspect_extractor loaded")


corpus_clusters = pd.read_csv("POC_clusters.csv")
st.write("corpus_clusters loaded")

###############################
st.write("# Update topic categories")
###############################

topic_categories = [
    'service',
    'hygeine', 
    'appetizer', 
    'desserts',
    'drinks/bar', 
    'food',
    'dishes', 
    'atmosphere', 
    'menu', 
    'price', 
    'management',
    'staff'
]
st.write(topic_categories)

def embed(model, sentences):
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    return embeddings

topic_categories_embeddings_st1 = embed(sentence_transformer, topic_categories)

clusters_embedding = embed(sentence_transformer, corpus_clusters.group.to_list())

from numpy.linalg import norm
def cosine_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine



labels = []
for i in range(len(clusters_embedding)):
    labels.append([cosine_sim(phrase, clusters_embedding[i]) for phrase in topic_categories_embeddings_st1])

test = pd.DataFrame(labels, columns =topic_categories)
test['topic'] = test.idxmax(axis=1)
test['topic_strength'] = test.max(axis=1)
test = test.round(2)

# test
st.write(test)


label_strength = pd.concat([corpus_clusters, test[['topic','topic_strength']]], axis=1)
st.write(label_strength)

label_strength_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic_strength']))
label_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic']))
label_dict[-1] = "Unknown"



user_review=['The cucumber cocktail was very refreshing, the pork in the hangover ramen was a bit hard to chew...',
            'The pistachio icecream was a delight! Service was ok']

atepc_result = aspect_extractor.extract_aspect(inference_source=user_review,
                                               save_result=False,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

for i in range(len(atepc_result)):
    atepc_result[i]['raw_text'] = user_review[i]

sentiment_dict = {'Positive': 1, 'Negative': -1, 'Neutral': 0}

cleanup = []
for i in atepc_result:
    for j in range(len(i['aspect'])):
        cleanup.append([
            i['raw_text'],
            i['aspect'][j],
                i['sentiment'][j],
            i['confidence'][j]
        ])
cleanup = pd.DataFrame(cleanup)
cleanup.columns = ['text', 'aspect', 'sentiment', 'confidence']
st.write(cleanup)

umap_reducer

test_new_points = cleanup['aspect'].tolist()
test_new_points_embeddings_st1 = embed(sentence_transformer, test_new_points)
test_new_points_embeddings_st1_umap = umap_reducer.transform(test_new_points_embeddings_st1)
test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusters, test_new_points_embeddings_st1_umap)

cleanup['topic'] = [label_dict[i] for i in test_labels]
cleanup['topic_strength'] = [label_strength_dict[i] for i in test_labels]


cleanup.confidence = cleanup.confidence.round(2)

df_display = cleanup.groupby(['text','sentiment']).agg(
    aspect=('aspect', lambda x: list(x)),
    aspect_topic=('topic', lambda x: list(x)),
    sentiment_confidence=('confidence', lambda x: list(x)),
    topic_confidence=('topic_strength', lambda x: list(x)),
)

st.write(df_display)