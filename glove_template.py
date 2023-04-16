#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 1
    count = 0

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            ##Do gradient decent on lossfuction J=fn*eta*(x.T*y-log(cooc_point))^2
            x=xs[ix,:]
            y=ys[jy,:]
            f=min((n/nmax)**alpha,1)
            xs[ix,:]=xs[ix,:]-2*f*y*(x.dot(y)-np.log(n))*eta
            ys[ix,:]=ys[jy,:]-2*f*x*(x.dot(y)-np.log(n))*eta
    np.save('embeddings', xs)


if __name__ == '__main__':
    main()
