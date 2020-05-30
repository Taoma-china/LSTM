from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras
import keras.utils as ku
import numpy as np
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.test.is_gpu_available())
# print(tf.test.gpu_device_name())
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(gpu_devices[0], True)
# print("GPUs: ", gpu_devices[0])
#
# gpus = tensorflow.test.gpu_device_name()
# print("GPUs: ", gpus)


tokenizer = Tokenizer(char_level=True)

def dataset_preparation(corpus):

    # basic cleanup

    # tokenization
    tokenizer.fit_on_texts(corpus)
    #sequence = tokenizer.texts_to_sequences(corpus)
    total_words = len(tokenizer.word_index) + 1

    #print("sequence: ", sequence, "\n")
    print("word_index: ", tokenizer.word_index, "\n")
    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    #max_sequence_len = max([len(x) for x in input_sequences])
    max_sequence_len=25
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words


def create_model(predictors, label, max_sequence_len, total_words,test_data,test_label):

    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
    model.add(LSTM(150, return_sequences=True,kernel_initializer=keras.initializers.he_uniform(seed=None)))
    # model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history_callback = model.fit(predictors, label, epochs=60, verbose=1,validation_data=(test_data, test_label))
    #after verbose, callbacks = [earlystop],
    print(model.summary())
    plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(history_callback.history['loss'], color='r', label='Training Loss')
    ax[0].plot(history_callback.history['val_loss'], color='g', label='test Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(history_callback.history['accuracy'], color='r', label='Training Accuracy')
    ax[1].plot(history_callback.history['val_accuracy'], color='g', label='test Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)

    plt.savefig("lstm_new.png")
    plt.show()
    return model


def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
corpus1 = corpus[:1000]
#corpus2=[]
#for i in range(0,100,5):
#    corpus2=corpus2+corpus[i:i+5]

predictors, label, max_sequence_len, total_words = dataset_preparation(corpus1)
#test_data, test_label, max_sequence_len, total_words = dataset_preparation(corpus2)
print(len(predictors))
print(len(label))
test_data = predictors[180000:]
test_label = label[180000:]

model= create_model(predictors, label, max_sequence_len, total_words,test_data,test_label)
model.save('lstm_new.h5')
print(generate_text("acdher", 3, max_sequence_len))

