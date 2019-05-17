from heapq import nlargest
from keras.preprocessing.text import text_to_word_sequence
import pickle

class BuildVocabulary:
    def __init__(self,dataset_path,vocabulary_path,num_of_words=30000):
        self.dataset_path = dataset_path
        self.num_of_words = num_of_words
        self.vocabulary_path = vocabulary_path

    def build(self):
        words_frequencies = dict()
        with open(self.dataset_path) as infile:
            for article in infile:
                words = text_to_word_sequence(article)
                for word in words:
                    if word not in words_frequencies:
                        words_frequencies[word]=0
                    else:
                        words_frequencies[word]+=1

        top_frequent_words = nlargest(self.num_of_words,words_frequencies,key=words_frequencies.get)

        vocabulary = dict(zip(top_frequent_words,range(1,len(top_frequent_words)+1)))

        pickle_out = open(self.vocabulary_path, "wb")
        pickle.dump(vocabulary, pickle_out)
        pickle_out.close()





