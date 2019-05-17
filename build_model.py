from keras import layers
from keras.models import  Model
from keras import backend as K
import tensorflow as tf
from tensorflow import Tensor
import numpy as np

class BuildModel:
    def __init__(self,n,n2,m,num_corrupt_examples,vocabulary_size):
        self.n = n
        self.n2 = n2
        self.m = m
        self.vocabulary_size = vocabulary_size
        self.num_corrupt_examples = num_corrupt_examples
        self.i =0

    def _get_last_vector(self,tensor):
        return tensor[:,self.n-1,:]

    def _get_first_vector(self,tensor):
        return tensor[:,self.i,:]

    def _margin_loss(self):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            f_vector = y_pred[:,0]
            sum=0
            for i in range(1,self.num_corrupt_examples):
                sum += K.maximum(0.0,1-f_vector+y_pred[:,i])

            return sum #K.mean(K.square(y_pred - y_true), axis=-1)

        # Return a function
        return loss

    def build(self):
        x1 = layers.Input((self.num_corrupt_examples,self.n,))
        x2 = layers.Input((self.n2,))

        instance_vector_layer = layers.Lambda(function=self._get_first_vector)

        embedding_1 = layers.Embedding(self.vocabulary_size,self.m)
        print ('Type of Embedding 1' + str(embedding_1))

        a1_layer = layers.Dense(100,activation='tanh', name='a1')
        scorel_l_layer = layers.Dense(1,activation=None)
        a1_g_layer = layers.Dense(100,activation='tanh')
        score_g_layer = layers.Dense(1,activation=None)

        #embedding_2 = layers.Embedding(self.vocabulary_size,self.m,input_length=self.n2)

        average_layer = layers.Lambda(lambda x: K.mean(x, axis=1))

        last_vector_layer = layers.Lambda(function=self._get_last_vector)

        concatenation_layer = layers.Concatenate()

        score_layer = layers.Add()


        previous_score = ""
        for self.i in range(0,self.num_corrupt_examples):

            instance_vector = instance_vector_layer(x1)
            encode = embedding_1(instance_vector)


            s = encode.get_shape().as_list()
            reshape = layers.Reshape((s[1]*s[2],))(encode)
            print ('Concatenate Embeddings shape' + str(reshape.shape))
            a1 = a1_layer(reshape)

            score_l = scorel_l_layer(a1)

            encode2 = embedding_1(x2)

            average = average_layer(encode2)

            last_vector = last_vector_layer(encode)
            conc = concatenation_layer([average,last_vector])
            a1_g = a1_g_layer(conc)

            score_g = score_g_layer(a1_g)

            score = score_layer([score_l,score_g])
            if self.i==0:
                previous_score = score
            else:
                previous_score = layers.concatenate([previous_score,score],axis=1)

        previous_score = previous_score

        model = Model(inputs=[x1, x2],outputs=previous_score)
        model.compile(loss=self._margin_loss(), optimizer="adam")

        model.summary()
        return model,embedding_1




