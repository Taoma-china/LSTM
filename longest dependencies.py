from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras.utils as ku
from collections import Counter
from random import choice
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
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        predicted_prob = model.predict(token_list,verbose=0)
        #print(predicted_prob)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text
def index_to_word(x):
    y=str()
    for i in range(len(x)):
        if x[i]==1:
            y=y+'L'
            y=y+' '
        if x[i]==2:
            y=y+'A'
            y=y+' '
        if x[i]==3:
            y=y+'G'
            y=y+' '
        if x[i]==4:
            y=y+'V'
            y=y+' '
        if x[i]==5:
            y = y + 'K'
            y=y+' '
        if x[i]==6:
            y = y + 'E'
            y=y+' '
        if x[i]==7:
            y = y + 'T'
            y=y+' '
        if x[i]==8:
            y = y + 'S'
            y=y+' '
        if x[i]==9:
            y = y + 'D'
            y=y+' '
        if x[i]==10:
            y = y + 'I'
            y=y+' '
        if x[i]==11:
            y = y +'R'
            y=y+' '
        if x[i]==12:
            y = y +'P'
            y=y+' '
        if x[i]==13:
            y = y +'N'
            y=y+' '
        if x[i]==14:
            y = y +'F'
            y=y+' '
        if x[i]==15:
            y = y +'Q'
            y=y+' '
        if x[i]==16:
            y = y +'Y'
            y=y+' '
        if x[i]==17:
            y = y +'H'
            y=y+' '
        if x[i]==18:
            y = y +'M'
            y=y+' '
        if x[i]==19:
            y = y +'C'
            y=y+' '
        if x[i]==20:
            y = y +'W'
            y=y+' '
    return y


data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
corpus1 = corpus[:1000]
#corpus2=[]
#for i in range(0,100,5):
#    corpus2=corpus2+corpus[i:i+5]
random_list = ['L','A','G','V','K','E','T','S','D','I','R','P','N','F','Q','Y','H','M','C','W']

predictors, label, max_sequence_len, total_words = dataset_preparation(corpus1)
test_data = corpus[500:1000]

model = load_model('lstm_new.h5')
#x=index_to_word(test_data[1][:13])
#x1=index_to_word(test_data[1])

correct=[]
all_correct=[]
test_corr=[]

print(test_data[0][:25])
for i in range(len(test_data)):

        if len(test_data[i])<25:
            continue
        #x=index_to_word(test_data[i])
        x_label=test_data[i][:25]
        x = test_data[i][:25]
        x.replace(' ','')
        #x_label.replace(' ','')
        for j in range(1,25):
            x=test_data[i][:25]
            judge=True


            x=list(x)
            x_compare = x[j-1]
            x[j-1]=choice(random_list)

            while(x_compare==x[j-1]):
                x[j-1]=choice(random_list)


            # print('xx[[i]]',x[i])
            print("testx[j]",x[:j])
            x_pre=generate_text(x[:j],25-j,25)
            #print('x_pre',x_pre)
            #print('x',x)
            x_pre=list(x_pre)
            new_pre=[]
            for w in range(len(x_pre)):
                if x_pre[w]!=' ':
                    new_pre.append(x_pre[w].upper())
            x_label=list(x_label)
            print('xla',(x_label))
            print('xpre',(new_pre))
            print("j",j)
            for k in range(len(new_pre)):
                if k==(j-1):
                    continue
                if x_label[k]!=new_pre[k]:
                    judge = False
            if (judge):
                correct.append(j)
                test_corr.append(j)
        all_correct.append(correct)
        correct=[]
print(all_correct)
print(Counter(test_corr))





print(generate_text("M N", 23, 25))# print(generate_text("M N I Y",21,25))

print(generate_text("M N I F E M L M", 17, 25))
print(generate_text("M N I F m M L M", 17, 25))
#print(generate_text("N",24,25))
# print(generate_text("M M",23,25))
#print(generate_text("MNL",22,25))# WU YING XIANG
# print(generate_text("M N I F Y",20,25))
# print(generate_text("M N I F E L",19,25))
# print(generate_text("M N I F E M N",18,25))
# print(generate_text("M N I F E M L E",17,25))#WU YING XIANG
# print(generate_text("M N I F E M L R G",16,25))
# print(generate_text("M N I F E M L R I L",15,25))
# print(generate_text("M N I F E M L R I D R",14,25))
# print(generate_text("M N I F E M L R I D E K",13,25))
# print(generate_text("M N I F E M L R G D E G I",12,25))
# print(generate_text("M N I F E M L R G D E G L K",11,25))
# print(generate_text("M N I F E M L R G D E G L R I",10,25))
# print(generate_text("M N I F E M L R G D E G L R L Y",9,25))
# print(generate_text("M N I F E M L R G D E G L R L K Y",8,25))
# print(generate_text("M N I F E M L R G D E G L R L K I K",7,25))
# print(generate_text("M N I F E M L R G D E G L R L K I Y D",6,25))
# print(generate_text("M N I F E M L R G D E G L R L K I Y K T",5,25))