import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from itertools import islice
import random

class KerasBatchGenerator:
    def __init__(self,dataset_path,n,n2,batch_size,vocabulay,num_corrupt_examples):
        self.dataset_path = dataset_path
        self.batch_size= batch_size
        self.vocabulary = vocabulay
        self.n = n
        self.n2 = n2
        self.num_corrupt_examples = num_corrupt_examples
        self.num_of_articles = 1

    def generate(self):
        while True:
            with open(self.dataset_path) as infile:
                self.num_of_articles = 1
                for article in infile:
                    print (article)
                    if self.num_of_articles % 2 == 0:
                        break
                    self.num_of_articles += 1
                    print ("WHEEEEE" + str(self.num_of_articles))
                    words = text_to_word_sequence(article)[:500]
                    indexes = []
                    for x in words:
                        if x in self.vocabulary:
                            indexes.append(self.vocabulary[x])
                        else:
                            indexes.append(0)

                    windows = []
                    windows.append([indexes[k:k+self.n] for k in range(len(indexes)-self.n-1)][:10])
                    for r in range(self.num_corrupt_examples-1):
                        windows.append([ indexes[k:k+self.n-1] + [self.vocabulary[random.choice(list(self.vocabulary.keys()))]] for k in range(len(indexes)-self.n-1)][:10])

                    document = indexes[:self.n2]
                    x1 = np.empty((len(windows[0]), self.num_corrupt_examples, self.n))
                    x2 = np.empty((len(windows[0]), self.n2))
                    y = np.empty((len(windows[0]), 1))
                    for s in range(len(windows[0])):
                        f=0
                        for j,z in enumerate(windows):
                            x1[s,j,:] = z[f]
                            y[s, 0] = 1  # don't used
                        f+=1

                        x2[s, : len(document)] = document
                    yield [x1, x2], y


    def _get_article(self):
        with open(self.dataset_path) as infile:
            self.num_of_articles = 1
            for article in infile:
                words = text_to_word_sequence(article)
                indexes = []
                for x in words:
                    if x in self.vocabulary:
                        indexes.append(self.vocabulary[x])
                    else:
                        indexes.append(0)
                windows = []
                windows.append([indexes[k:k+self.n-1] for k in range(len(indexes)-self.n-1)])
                yield article


    def __window(self,seq, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result


