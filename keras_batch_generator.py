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
        context_generator = self._get_article()
        while True:
            self.num_of_articles = 1
            x1 = np.empty((self.batch_size,self.num_corrupt_examples, self.n))
            x2 = np.empty((self.batch_size,self.n2))
            y = np.empty((self.batch_size,1))
            for i in range(self.batch_size):
                a,b,c =  next(context_generator)
                x1[i,:] = a
                x2[i,:] = b
                y[i,:] = c

            yield [x1,x2],y


    def _get_article(self):
        with open(self.dataset_path) as infile:
            self.num_of_articles = 1
            for article in infile:
                words = text_to_word_sequence(article)[:200]
                print (words)
                indexes = []
                for x in words:
                    if x in self.vocabulary:
                        indexes.append(self.vocabulary[x])
                    else:
                        indexes.append(0)
                windows = []
                windows.append([indexes[k:k+self.n-1] for k in range(len(indexes)-self.n-1)][:10])
                for r in range(self.num_corrupt_examples - 1):
                    windows.append(
                        [indexes[k:k + self.n - 1] + [self.vocabulary[random.choice(list(self.vocabulary.keys()))]] for
                         k in range(len(indexes) - self.n - 1)][:10])


                document = indexes[:self.n2]
                for s in range(len(windows[0])):
                    x1 = np.empty((self.num_corrupt_examples, self.n))
                    x2 = np.empty((self.n2))
                    y = np.empty((1))
                    f = 0
                    for j, z in enumerate(windows):
                        x1[j, : len(z[f])] = z[f]

                    y[0] = 1  # don't used
                    f += 1
                    x2[: len(document)] = document
                    yield x1,x2, y


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


