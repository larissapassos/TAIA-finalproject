from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from ast import literal_eval as eval
from nltk.corpus import stopwords
from twokenize import tokenize as tweet_tokenizer
import Processor

def main():
    svm = LinearSVC()
    lines = [eval(line.strip()) for line in open("tweetlabels.txt", "r")]
    stop = stopwords.words("english")
    stop.remove("no")
    corpus = [x[0] for x in lines]
    labels = [x[1] for x in lines]
    pr = Processor.Processor(stop)
    normalized_corpus, lines = pr.process(corpus)
    del lines
    print normalized_corpus[:10]
    tfidf = Tfidf(ngram_range=(1,2), binary=True, tokenizer=tweet_tokenizer)
    X = tfidf.fit_transform(normalized_corpus)

if __name__ == '__main__':
    main()
