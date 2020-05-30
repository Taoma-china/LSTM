from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras.utils as ku
from collections import Counter
from random import choice
import xlwt
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


f = xlwt.Workbook()
sheet1 = f.add_sheet("result",cell_overwrite_ok=True)

row0 =[" ","0","1","2","3",'4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19+']
row1 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row2 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row3 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row4 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row5 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row6 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row7 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row8 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row9 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
row10 = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']

colum0=['1','2','3','4','5','6','7','8','9','10']
for i in range(0,len(row0)):
    sheet1.write(0,i,row0[i])
for i in range(0,len(colum0)):
    sheet1.write(i+1,0,colum0[i])



data = open('pdb_seqres.txt').read()
corpus = data.split("\n")
corpus1 = corpus[:1000]
#corpus2=[]
#for i in range(0,100,5):
#    corpus2=corpus2+corpus[i:i+5]
random_list = ['L','A','G','V','K','E','T','S','D','I','R','P','N','F','Q','Y','H','M','C','W']

predictors, label, max_sequence_len, total_words = dataset_preparation(corpus1)
test_data = corpus[950:951]

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
        print('xla', (x_label))
        #x_label.replace(' ','')
        for j in range(1,11):
            x=test_data[i][:25]
            judge=True


            x=list(x)



            # print('xx[[i]]',x[i])
            print("fix k sequence",x[:j])
            x_pre=generate_text(x[:j],25-j,25)
            #print('x_pre',x_pre)
            #print('x',x)
            x_pre=list(x_pre)
            new_pre=[]
            for w in range(len(x_pre)):
                if x_pre[w]!=' ':
                    new_pre.append(x_pre[w].upper())
            x_label=list(x_label)

            print('xpre',(new_pre))
            print("fix k:",j)
            for k in range(j+1,len(new_pre)):

                if x_label[k]!=new_pre[k]:

                    break
            length = k - j - 1
            if length>19:
                length=19

            if j ==1:
                row1[length]=1

            if j ==2:
                row2[length]=1
            if j == 3:
                row3[length] = 1
            if j == 4:
                row4[length] = 1
            if j == 5:
                row5[length] = 1
            if j == 6:
                row6[length] = 1
            if j == 7:
                row7[length] = 1
            if j == 8:
                row8[length] = 1
            if j == 9:
                row9[length] = 1
            if j == 10:
                row10[length] = 1


for i in range(0,len(row1)):
    sheet1.write(1,i+1,row1[i])

for i in range(0,len(row2)):
    sheet1.write(2,i+1,row2[i])

for i in range(0,len(row3)):
    sheet1.write(3,i+1,row3[i])
for i in range(0,len(row4)):
    sheet1.write(4,i+1,row4[i])
for i in range(0,len(row5)):
    sheet1.write(5,i+1,row5[i])
for i in range(0,len(row6)):
    sheet1.write(6,i+1,row6[i])
for i in range(0,len(row7)):
    sheet1.write(7,i+1,row7[i])
for i in range(0,len(row8)):
    sheet1.write(8,i+1,row8[i])
for i in range(0,len(row9)):
    sheet1.write(9,i+1,row9[i])
for i in range(0,len(row10)):
    sheet1.write(10,i+1,row10[i])


f.save('rere.xls')