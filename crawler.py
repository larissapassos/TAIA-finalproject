import twitter
import time
import cPickle as pickle

def main():
    keys = [key.strip() for key in open("keys.txt", "r")]
    api = twitter.Api(keys[0], keys[1], keys[2], keys[3])
    tweets_labels = []
    i = 0
    t0 = time.time()
    with open("tweetlabels.txt", "a") as outfile:
        for tweet in api.GetStreamFilter(track=[":)", ":("]):
            if tweet.has_key("lang") and tweet.has_key("text") and \
                                              tweet["lang"] == "en":
                text = tweet["text"]
                label = 1 if ":)" in text else -1 if ":(" in text else 0
                if label != 0:
                    tweet_label = str((text, label))+"\n"
                    outfile.write(tweet_label)
                    i += 1
                    if i % 50 == 0:
                        t1 = time.time() - t0
                        print "tweets: %d, time elapsed: %.0fs" % (i, t1)
                    if i == 10000:
                        break


if __name__ == '__main__':
    main()
