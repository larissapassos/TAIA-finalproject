from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from ast import literal_eval as eval
from nltk.corpus import stopwords
from twokenize import tokenize as tweet_tokenizer
import Processor, sys
import numpy as np

def process_corpus():
    lines = [eval(line.strip()) for line in open("tweetlabels.txt", "r")]
    stop = stopwords.words("english")
    stop.remove("no")
    corpus = [x[0] for x in lines]
    labels = [x[1] for x in lines]
    pr = Processor.Processor(stop)
    normalized_corpus, lines = pr.process(corpus)
    del lines
    tfidf = Tfidf(ngram_range=(1,2), binary=True,
                  tokenizer=tweet_tokenizer)
    X = tfidf.fit_transform(normalized_corpus)
    return (X, labels)

def eval_params(X, labels, C, gamma):
    cutoff = int(X.shape[0]*.7)
    svm = SVC(C=C, gamma=gamma)
    X_train, y_train = X[:cutoff], labels[:cutoff]
    X_test, y_test = X[cutoff:], np.array(labels[cutoff:])
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    fitness = (preds == y_test).sum() / (y_test.shape[0]+.0)
    return fitness

def main(C, gamma):
    X, labels = process_corpus()
    print eval_params(X, labels, C, gamma)

if __name__ == '__main__':
    C = float(sys.argv[1])
    gamma = float(sys.argv[2])
    main(C, gamma)
