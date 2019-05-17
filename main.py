from keras.callbacks import ModelCheckpoint
from build_vocabulary import BuildVocabulary
from build_model import BuildModel
from keras_batch_generator import KerasBatchGenerator
import os,pickle


num_of_articles = 200000 #4648219 # 8

dataset_path = "dataset/en-wiki-clean-text.txt"
vocabulary_path = "dataset/vocabulary.pkl"


if not os.path.isfile(vocabulary_path):
    vocabulary_builder = BuildVocabulary(dataset_path,vocabulary_path)
    vocabulary_builder.build()


pickle_in = open(vocabulary_path,"rb")
vocabulary = pickle.load(pickle_in)



n = 10
n2 = 100
m = 50
num_corrupt_examples = 10

model_builder =BuildModel(n,n2,m,num_corrupt_examples,len(vocabulary))
model,embedding_layer = model_builder.build()

batch_size = 1

checkpointer = ModelCheckpoint(filepath='model-{epoch:02d}.hdf5', verbose=1)



# train_text,valid_text = load_dataset()
train_data_generator = KerasBatchGenerator(dataset_path,n,n2,batch_size,vocabulary,num_corrupt_examples)
# valid_data_generator = KerasBatchGenerator(n,n2,batch_size,vocabulary)



results = model.fit_generator(train_data_generator.generate(), num_of_articles // batch_size,
                                   epochs=1,
                                   callbacks=[checkpointer],
                                 )

embedding_matrix = embedding_layer.get_weights()
model.save_weights('dataset/embedding_matrix_weights.h5')
# print(np.mean(results.history["val_acc"]))

#                                   validation_data=valid_data_generator.generate(),
#                                   validation_steps=len(valid_text)*n2 // batch_size,