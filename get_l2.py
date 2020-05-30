
#
import numpy as np
#

def get_L2_distance(vect1, vect2):
    dist = np.sqrt(np.sum(np.square(vect1 - vect2)))
    # 或者用numpy内建方法
    # dist = numpy.linalg.norm(vect1 - vect2)
    return dist


#
# vect1 = np.array(['A','B','C'])
# vect2 = np.array(['D', 'W', 'F'])

# print(get_edclidean_distance(vect1, vect2))

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras.utils as ku
from collections import Counter
from random import choice
tokenizer = Tokenizer(char_level=True)

def get_index(x):
    if x =='L':
        return 0
    if x =='A':
        return 1
    if x =='G':
        return 2
    if x =='V':
        return 3
    if x =='K':
        return 4
    if x =='E':
        return 5
    if x =='T':
        return 6
    if x =='S':
        return 7
    if x =='D':
        return 8
    if x =='I':
        return 9
    if x =='R':
        return 10
    if x =='P':
        return 11
    if x =='N':
        return 12
    if x =='F':
        return 13
    if x =='Q':
        return 14
    if x =='Y':
        return 15
    if x =='H':
        return 16
    if x =='M':
        return 17
    if x =='C':
        return 18
    if x =='W':
        return 19


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
def generate_text(seed_text, next_words, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    #print("tokem_listss",token_list)
    predicted = model.predict_classes(token_list, verbose=0)
    predicted_prob = model.predict(token_list,verbose=0)
    #print('TESTTT',predicted_prob)

    return predicted_prob



data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
corpus1 = corpus[:1000]
#corpus2=[]
#for i in range(0,100,5):
#    corpus2=corpus2+corpus[i:i+5]
random_list = ['L','A','G','V','K','E','T','S','D','I','R','P','N','F','Q','Y','H','M','C','W']

predictors, label, max_sequence_len, total_words = dataset_preparation(corpus1)
test_data = corpus[:1]

#sample_800 = test_data[800][0:25]
#print(sample_800)

model = load_model('lstm_new.h5')

vector1 = []
vector2 = []
#MQNGYTYEDYQDTAKWLLSHTEQRP
#MQNG
for i in range(22):

    #print("est_data",test_data)
    test = test_data[0][i:i+4]
    #print("TEST",test)
    print("number:",i)
    P_m = generate_text('',25,25)
    index = get_index(test[0])
    #print(index)
    P_m=P_m[0][index]
    print("P(first)",P_m)
    vector1.append(P_m)

    P_qM = generate_text(test[0],24,25)
    index=get_index(test[1])
    P_qM=P_qM[0][index]
    print("P(second|first)",P_qM)
    vector1.append(P_qM)

    P_nMQ=generate_text(test[0:2],23,25)
    index=get_index((test[2]))
    P_nMQ=P_nMQ[0][index]
    print("P(third|first two)",P_nMQ)
    vector1.append(P_nMQ)

    P_gNMQ=generate_text(test[0:3],22,25)
    index=get_index(test[3])
    P_gNMQ=P_gNMQ[0][index]
    print("P(forth|first three)",P_gNMQ)
    vector1.append(P_gNMQ
                   )
    P_mqng = P_m * P_qM * P_nMQ * P_gNMQ
    print("P(all four)",P_mqng)
    vector1.append(P_mqng)
print("********************************4-gram-MODEL***********************************")
#
model = load_model('lstm_4gram.h5')
#
for i in range(22):
    #print("est_data",test_data)
    test = test_data[0][i:i+4]
    #print("TEST",test)
    print("number:",i)
    P_m = generate_text('',25,25)
    index = get_index(test[0])
    #print(index)
    P_m=P_m[0][index]
    print("P(first)",P_m)
    vector2.append(P_m)

    P_qM = generate_text(test[0],24,25)
    index=get_index(test[1])
    P_qM=P_qM[0][index]
    print("P(second|first)",P_qM)
    vector2.append(P_qM)

    P_nMQ=generate_text(test[0:2],23,25)
    index=get_index((test[2]))
    P_nMQ=P_nMQ[0][index]
    print("P(third|first two)",P_nMQ)
    vector2.append(P_nMQ)

    P_gNMQ=generate_text(test[0:3],22,25)
    index=get_index(test[3])
    P_gNMQ=P_gNMQ[0][index]
    print("P(forth|first three)",P_gNMQ)
    vector2.append(P_gNMQ)

    P_mqng = P_m * P_qM * P_nMQ * P_gNMQ
    print("P(all four)",P_mqng)
    vector2.append(P_mqng)

print(np.array(vector1))
print(np.array(vector2))
L2_list=[]
for i in range(22):
    vector_com1=np.array(vector1[i:i+5])
    vector_com2=np.array(vector2[i:i+5])
    result=get_L2_distance(vector_com1,vector_com2)
    L2_list.append(result)
print(L2_list)


test_list =[0.039207775, 0.04069929, 0.04001626, 0.018126749, 0.019602427, 0.019602427, 0.08058691, 0.081122674, 0.08115459, 0.08123663, 0.08123663, 0.021891603, 0.057534095, 0.30106863, 0.30093846, 0.30093846, 0.30048206, 0.30135483, 0.05976149, 0.059761565, 0.059761565, 0.062464472,0.38184065, 0.13414468, 0.03431995, 0.027517552, 0.5201075, 0.5201075, 0.5199882, 0.5209995, 0.59083956, 0.28165892, 0.28165892, 0.28178003, 0.28209049, 0.043571703, 0.3949226, 0.3949226, 0.3957952, 0.40030295, 0.40036166, 0.079655625, 0.079655625, 0.08848169,0.019436693, 0.051582653, 0.050369024, 0.35893875, 0.35886016, 0.35886016, 0.35538715, 0.35934687, 0.054046277, 0.06747922, 0.06747922, 0.3643123, 0.37322173, 0.37348044, 0.4040606, 0.4040606, 0.20279959, 0.19148451, 0.571188, 0.5484942, 0.5484942, 0.5432493]
test_list.sort()