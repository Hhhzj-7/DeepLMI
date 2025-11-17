
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
import torch.nn.functional as F
from tqdm import tqdm


def k_mers(k, seq):
    if k > len(seq):
        return []

    num = len(seq)-k+1
    split = []
    for i in range(num):
        split.append(seq[i:i+k])

    return split


def create_tagged_docs(mers, name):
    tagged_docs = [TaggedDocument(mers[i], name[i])
                   for i in range(len(name))]

    return tagged_docs


def train_doc2vec(mers, name):
    tagged_docs = create_tagged_docs(mers, name)
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model


def get_vector(all_mers, all_name, model):
    tagged_docs = create_tagged_docs(all_mers, all_name)
    vectors = np.array([])
    vectors = {}
    for doc in tqdm(tagged_docs, desc="Inferring embeddings"):
        vectors[doc.tags] = model.infer_vector(doc.words)

    return vectors

def get_dcg(y_pred, y_true, k):
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    df = df.sort_values(by="y_pred", ascending=False)
    df = df.iloc[0:k, :]
    dcg = (2 ** df["y_true"] - 1) / \
        np.log2(np.arange(1, df["y_true"].count() + 1) + 1)
    dcg = np.sum(dcg)
    return dcg

def NDCG(y_true, y_pred):
    k = len(y_pred)
    dcg = get_dcg(y_pred, y_true, k)
    idcg = get_dcg(y_true, y_true, k)
    ndcg = dcg / idcg
    return ndcg

