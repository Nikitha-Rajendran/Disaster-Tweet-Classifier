import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from keras.preprocessing import text, sequence
from keras.layers import Activation,Dense, Embedding, Dropout, LSTM, Input
from keras.models import  Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df=pd.read_csv(r"train_file_path" ,usecols = ["text","target"])

df.text

sns.set_theme(style='darkgrid')
sns.countplot(df.target)
plt.ylabel('no. of tweets')
plt.xlabel('disaster or not')
plt.title('tweets')

def clean(tweet):
    tweet=tweet.lower()
    tweet=re.sub(r"http\S+"," ",tweet)
    tweet= re.sub(r'\W'," ",tweet)  
    tweet=re.sub(r'[0-9]'," ",tweet)
    return tweet
cleaning=lambda x: clean(x)
df['text']=df['text'].apply(cleaning)

df

#stopwords
f=open(r"stopwords_file_path",encoding="utf-8")
content = f.read()
stopwords = content.split(",")
f.close()
stopwords=[i.replace('"',"").strip() for i in stopwords]

df['text']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

df

def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

df['text'] = get_lemmatized_text(df['text'])

df

x_train=df.text
y_train=df.target

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(5,25))
train_len=df[df['target']==1]['text'].str.split().map(lambda x:len(x))
ax1.hist(train_len,color='blue')
ax1.set_title('disaster')

train_len=df[df['target']==0]['text'].str.split().map(lambda x:len(x))
ax2.hist(train_len,color='orange')
ax2.set_title('not a disaster')
print('no of words')
plt.show()

max_len=30
max_words=10000
tok=text.Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences=tok.texts_to_sequences(x_train)
sequences_matrix=sequence.pad_sequences(sequences,maxlen=max_len,padding='post')

embedding_index=dict()
#f=open(r"gensim_file_path",encoding="Latin-1")
f1=open(r'glove_embeddings_file_path',encoding="utf-8") #https://stackoverflow.com/questions/9233027/unicodedecodeerror-charmap-codec-cant-decode-byte-x-in-position-y-character
for line in f1:
    values=line.split()
    word=values[0]
    coefs=np.array(values[1:])
    embedding_index[word]=coefs
f1.close()
k=len(embedding_index)
print(f'Loaded {k} word vectors')

vocab_size_train=len(tok.word_index)+1
embedding_matrix=np.zeros((vocab_size_train,100))
for word,i in tok.word_index.items():
    embedding_vector=embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

print(vocab_size_train)

def RNN():
    inputs=Input(name='inputs',shape=[max_len])
    layer=Embedding(vocab_size_train,100,weights=[embedding_matrix],input_length=30,trainable=False)(inputs)
    layer=LSTM(64)(layer)
    layer=Dense(128)(layer)
    layer=Activation('relu')(layer)
    layer=Dropout(0.5)(layer)
    layer=Dense(64)(layer)
    layer=Activation('relu')(layer)
    layer=Dropout(0.5)(layer)
    layer=Dense(1)(layer)
    layer=Activation('sigmoid')(layer)
    model=Model(inputs=inputs,outputs=layer)
    return model

model=RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

model.fit(sequences_matrix,y_train,batch_size=512,epochs=100,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.00001)])

dt=pd.read_csv(r"test_file_path")

dt.text=dt.text.apply(cleaning)
dt['text']=dt['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
dt['text'] = get_lemmatized_text(dt['text'])
X_test=dt.text

dt

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

Y_test = model.predict(test_sequences_matrix)

dt1=pd.DataFrame(dt.id)

dt1['target']=Y_test.round().astype(int)

dt1.to_csv('disas_submission6.csv', index=False)


