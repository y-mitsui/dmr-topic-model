from __future__ import print_function
import pandas as pd
from scipy.sparse import coo_matrix
from gensim.matutils import corpus2csc
import dmr
import sys
import numpy as np
import pickle

if __name__ == "__main__":
    n_top_movies = 1000
    ratings = pd.read_csv("data/ratings.csv")
    sort_voting = sorted(ratings.groupby("movieId")["userId"].count().to_dict().items(), key=lambda x:x[1], reverse=True)
    enable_movies = dict(zip([item[0] for item in sort_voting[:n_top_movies]], [True] * n_top_movies))
    ratings = ratings[[enable_movies.get(m_id) != None for m_id in ratings["movieId"]]]
    movies = pd.read_csv("data/movies.csv")
    movie2title = dict(zip(movies["movieId"], movies["title"]))
    
    unique_item = ratings["movieId"].drop_duplicates()
    item2idx = dict(zip(unique_item, range(unique_item.shape[0])))
    idx2item = {v:k for k, v in item2idx.items()}
    unique_user = ratings["userId"].drop_duplicates()
    user2idx = dict(zip(unique_user, range(unique_user.shape[0])))
    
    corpus = [[] for _ in range(unique_item.shape[0])]
    for _, row in ratings.iterrows():
        corpus[item2idx[row["movieId"]]].append((user2idx[row["userId"]], row["rating"]))
        
    movies = pd.read_csv("data/movies.csv")
    genres = []
    for genre in movies["genres"]:
        genres.extend(genre.split("|"))
    unique_genre = pd.Series(genres).drop_duplicates()
    genre2idx = dict(zip(unique_genre, range(unique_genre.shape[0])))
    
    features = [float('nan')] * unique_item.shape[0]
    id2title = {}
    for _, row in movies.iterrows():
        if item2idx.get(row["movieId"]) is None:
            continue
            
        bow = [0] * unique_genre.shape[0]
        for genre in row["genres"].split("|"):
            bow[genre2idx[genre]] = 1.
            
        features[item2idx[row["movieId"]]] = bow
        id2title[row["movieId"]] = row["title"]
        
    if np.any(np.isnan(features)):
        raise Exception("error features")
        
    print("matrix size:%d X %d"%(unique_item.shape[0], unique_user.shape[0]))
    print("feature dimentions:%d"%(unique_genre.shape[0]))
    
    if True:
        weights = np.array([1.] * len(features))
        model = dmr.LDA(20, 81)
        item_user = corpus2csc(corpus).T.astype(int)
        model.fit(item_user, np.array(features), weights)
        
        with open("data/dmr.p", "wb") as fh:
            pickle.dump(model, fh)
    else:
        with open("data/dmr.p", "rb") as fh:
            model = pickle.load(fh)
        
    W = model.doc_topic_
    H = model.topic_word_
    
    log_W = np.log(W)
    for i in [item2idx[m_id] for m_id, _ in sort_voting[:n_top_movies]]:
        distances = np.sum(W[i] * (log_W[i] - log_W), 1)
        sort_idx = np.argsort(distances)
        recom = ",".join([id2title[idx2item[idx]] for idx in sort_idx[sort_idx != i][:5]])
        print("%s,%s"%(id2title[idx2item[i]], recom))
            
        
        
        
        
        
        
        
