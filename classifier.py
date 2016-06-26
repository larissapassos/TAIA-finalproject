from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from ast import literal_eval as eval
from nltk.corpus import stopwords
from twokenize import tokenize as tweet_tokenizer
import Processor, sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

class ML:
    def __init__(self):
        self.X = None
        self.labels = None

    def process_corpus(self):
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
        self.X = X
        self.labels = labels

    def eval_params(self, nparr):
        cutoff = int(self.X.shape[0]*.7)
        C = nparr[0]
        gamma = nparr[1]
        if gamma < 0:
            gamma = 1e-3
        if C < 0:
            C = 1e-3
        svm = SVC(C=C, gamma=gamma)
        X_train, y_train = self.X[:cutoff], self.labels[:cutoff]
        X_test, y_test = self.X[cutoff:], np.array(self.labels[cutoff:])
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        # fitness = (y_pred == y_test).sum() / (y_test.shape[0]+.0)
        fitness = prfs(y_test, y_pred, average='macro')[2]
        print "F-score macro: %.6f achieved with C=%.6f and gamma=%.6f" % (fitness, C, gamma)
        return fitness
