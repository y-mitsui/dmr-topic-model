from __future__ import print_function
import pandas as pd
from gensim.matutils import corpus2csc, corpus2dense
import dmr
import sys
import numpy as np
import pickle
from scipy.spatial.distance import cdist

def get_rating_error(r, p, q):
    return r - np.dot(p, q)

def get_error(R, P, Q, beta):
    error = 0.0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] == 0:
                continue
            error += pow(get_rating_error(R[i][j], P[:,i], Q[:,j]), 2)
    error += beta/2.0 * (np.linalg.norm(P) + np.linalg.norm(Q))
    return error

def matrix_factorization(R, K, steps=1000, alpha=0.0002, beta=0.02, threshold=0.001):
    P = np.random.rand(K, len(R))
    Q = np.random.rand(K, len(R[0]))
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] == 0:
                    continue
                err = get_rating_error(R[i][j], P[:, i], Q[:, j])
                for k in xrange(K):
                    P[k][i] += alpha * (2 * err * Q[k][j])
                    Q[k][j] += alpha * (2 * err * P[k][i])
        error = get_error(R, P, Q, beta)
        if error < threshold:
            break
        if (step + 1) % max(int(steps / 100), 1) == 0:
            print("%d / %d %.4f"%(step + 1, steps, error))
    return P.T, Q
    
if __name__ == "__main__":
    n_top_movies = 500
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
    
    if False:
        item_user = corpus2dense(corpus, len(user2idx)).T.astype(int)
        W, H = matrix_factorization(item_user, 20, steps=150, alpha=0.002)
        for i in [item2idx[m_id] for m_id, _ in sort_voting[:n_top_movies]]:
            distances = cdist([W[i]], W)[0]
            sort_idx = np.argsort(distances)
            recom = ",".join([id2title[idx2item[idx]] for idx in sort_idx[sort_idx != i][:5]])
            print("%s,%s"%(id2title[idx2item[i]], recom))
        sys.exit(0)
    
    if True:
        weights = np.array([1.] * len(features))
        model = dmr.LDA(30, 81)
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
            
        
        
        
        
        
        
        
