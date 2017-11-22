# -*- coding: utf-8 -*-
import dmr
from gensim import corpora, models, similarities
from gensim.matutils import corpus2csc
import numpy as np


np.random.seed(12345)
documents = [
             "LSI LDA 手軽 試せる gensim 使った 単語 自然 言語 処理 入門",
             "単語 ベクトル 化 する word2vec gensim LDA 使い 指定 二 単語 間 関連",
             "word2vec 仕組み gensim 使う 文書 類似 度 算出 チュートリアル",
             "機械学習 これ 始める 人 押さえる ほしい こと",
             "初心者 向け 機械学習 ディープラーニング 違い シンプル 解説",
             "機械学習 データサイエンティスト 機械学習 ディープラーニング エンジニア なる スキル 要件",
             "セクハラ やじ 浴びた 前 都議 民進党 衆院 選 ",
             "執行部 成立 させる なくなる 民進党 内ゲバ 離党 ドミノ 衆院 選",
             "前原 代表 選 民進党 再生 できる",
             "機械学習 gensim",
             ]
true_data = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
], dtype=float)

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in documents]
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
            for text in texts]
dictionary = corpora.Dictionary(texts)
new_doc = "Human computer interaction"
corpus = [np.array(dictionary.doc2bow(text)) for text in texts]
n_topics = 3
user_item_matrix = corpus2csc(corpus).T
user_item_matrix = user_item_matrix.astype(int)
ctx = lda.LDA(n_topics, 100)
doc_topics = ctx.fit_transform(user_item_matrix, true_data)
print(np.argmax(doc_topics, 1))
