import nltk
import numpy as np
import re
import pandas
import string
import os
import tqdm

from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


'''
df=pd.concat([tweet,test])
df.shape


df['text']=df['text'].apply(lambda x : remove_URL(x))


df['text']=df['text'].apply(lambda x : remove_html(x))

df['text']=df['text'].apply(lambda x: remove_emoji(x))

df['text']=df['text'].apply(lambda x : remove_punct(x))

'''
class Cleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def change_dataset(self, dataset):
        self.dataset = dataset

    def clean(self):
        for i, text in enumerate(self.dataset):
            text = self.remove_URL(text)
            text = self.remove_html(text)
            text = self.remove_emoji(text)
            text = self.remove_punct(text)
            ###  text = text.lower()
            text = self.remove_stopwords(text)
            
            self.dataset[i] = text

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        text = url.sub(r'',text)

        return text
    
    def remove_html(self, text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)
    
    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_punct(self, text):
        table=str.maketrans('','',string.punctuation)
        return text.translate(table)
    
    def remove_stopwords(self, text):
        text = text.split()
        text = [word for word in text if not word in stop_words]
        return text

train_path = 'dataset/train.csv'
test_path = 'dataset/test.csv'
dataset_train = pandas.read_csv(train_path, dtype={'id':'int', 'keyword':'str', 'location':'str', 'text':'str', 'target':'int'})
train_x = dataset_train['text']
train_y = dataset_train['target']

N_real_disaster = train_y.sum()
N_fake_disaster = len(train_y) - N_real_disaster
print(N_real_disaster)
print(N_fake_disaster)

dataset_test = pandas.read_csv(test_path, dtype={'id':'int', 'keyword':'str', 'location':'str', 'text':'str'})
test_x = dataset_test['text']

cleaner = Cleaner(train_x)

cleaner.clean()

cleaner.change_dataset(test_x)

cleaner.clean()

class BagOfWords:
    def __init__(self, dataset):
        self.dataset = dataset
        self.word_dict = {}

    def create_bag_of_words(self):
        for line in self.dataset:
            #tokens = nltk.word_tokenize(line)
            for token in line:
                self.word_dict[token] = 1 if token not in self.word_dict.keys() else self.word_dict[token] + 1
    
    def add_words_to_bag(self, test_set):
        for line in test_set:
            for token in line:
                self.word_dict[token] = 1 if token not in self.word_dict.keys() else self.word_dict[token] + 1

bag_of_words = BagOfWords(train_x)
bag_of_words.create_bag_of_words()
bag_of_words.add_words_to_bag(test_x)



naive_bayes_model = MultinomialNB(alpha=1.0, fit_prior=True)

x_train, x_validation, y_train, y_validation = train_test_split( train_x, train_y, test_size = 0.2)

naive_bayes_model.fit(x_train, y_train)
print( naive_bayes_model.score(x_validation, y_validation))

'''
clf = MultinomialNB(alpha=1.0, fit_prior=True)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

'''



'''
┈┈╱╲┈┈┈╱╲┈┈╭━╮┈
┈╱╱╲╲__╱╱╲╲┈╰╮┃┈
┈▏┏┳╮┈╭┳┓▕┈┈┃┃┈
┈▏╰┻┛▼┗┻╯▕┈┈┃┃┈
┈╲┈┈╰┻╯┈┈╱▔▔┈┃┈
┈┈╰━┳━━━╯┈┈┈┈┃┈
┈┈┈┈┃┏┓┣━━┳┳┓┃┈
┈┈┈┈┗┛┗┛┈┈┗┛┗┛┈
'''
