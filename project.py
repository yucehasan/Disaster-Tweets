import nltk
import numpy as np
import re
import pandas
import string
import os
import tqdm
import csv
import matplotlib.pyplot as plt
from operator import itemgetter 

from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

'''
df=pd.concat([tweet,test])
df.shape
df['text']=df['text'].apply(lambda x : remove_URL(x))
df['text']=df['text'].apply(lambda x : remove_html(x))
df['text']=df['text'].apply(lambda x: remove_emoji(x))
df['text']=df['text'].apply(lambda x : remove_punct(x))
'''

def generate_feature_set( train_X, test_X, vocab_list):
    #vocab_list is a dictionary, the others are numpy arrays
    feat_counts_train = np.zeros( (train_X.shape[0], len(vocab_list)))
    feat_counts_test = np.zeros( (test_X.shape[0], len(vocab_list)))
    #two separate loops, one for training set and the other for the test set
    for index, row in enumerate( train_X):
        np_row = np.array( row)
        counted = np.empty((len(row)), dtype='str')
        for word_i, word in enumerate( np_row):
            if word not in counted:
                count = np.count_nonzero( np_row == word)
                feat_counts_train[ index][ vocab_list.index(word)] += count
                counted[word_i] = word
    feat_counts_train = feat_counts_train.astype(int)
    
    for index, row in enumerate( test_X):
        np_row = np.array( row)
        counted = np.empty((len(row)), dtype='str')
        for word_i, word in enumerate( np_row):
            if word not in counted:
                count = np.count_nonzero( np_row == word)
                feat_counts_test[ index][ vocab_list.index(word)] += count
                counted[word_i] = word
    feat_counts_test = feat_counts_test.astype(int)

    return feat_counts_train, feat_counts_test

def predict_for_test_set( test_X, test_file_name, nb_model):
    with open(test_file_name, 'w') as file:
        writer = csv.writer(file)

        prediction_list = []

        predictions_array = nb_model.predict_proba(test_X)
        predictions_in_class = np.argmax( predictions_array, axis=1).reshape(-1,1)
        writer.writerows( predictions_in_class)

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
            text = text.lower()
            text = self.remove_stopwords(text)
            text = self.remove_self_defined_tokens(text)
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

    def remove_self_defined_tokens(self, text):
        path = "self_defined.csv"
        self_defined = list(pandas.read_csv(path, dtype={'words': 'string'})["words"])

        text = [word for word in text if not word in self_defined]
        return text

class BagOfWords:
    def __init__(self, dataset):
        self.dataset = dataset
        self.word_dict = {}
        self.word_indices = []
        self.vocab_list = []

    def create_bag_of_words(self):
        for line in self.dataset:
            #tokens = nltk.word_tokenize(line)
            for token in line:
                self.word_dict[token] = 1 if token not in self.word_dict.keys() else self.word_dict[token] + 1
                if token not in self.word_indices:
                    self.word_indices.append(token)
    
    def add_words_to_bag(self, test_set):
        for line in test_set:
            for token in line:
                self.word_dict[token] = 1 if token not in self.word_dict.keys() else self.word_dict[token] + 1
                if token not in self.word_indices:
                    self.word_indices.append(token)
    
    def convert_to_list(self):
        for key, value in self.word_dict.items():
            self.vocab_list.append(key)

class Grapher:
    def __init__(self, x_axis, y_axis, plot_type, title = ""):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title
        self.plot_type = plot_type

    def change_plot_type(self, plot_type):
        self.plot_type = plot_type
    
    def change_x_axis(self, x_axis):
        self.x_axis = x_axis

    def change_y_axis(self, y_axis):
        self.y_axis = y_axis

    def change_axises(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
    
    def change_title(self, title):
        self.title = title
    
    def plot(self, first_y_axis = [], second_y_axis = [], first_label = '', second_label = ''):
        plt.title(self.title)

        if self.plot_type == "bar":
            plt.bar(self.x_axis, self.y_axis)
            plt.show()
        elif self.plot_type == "grouped_bar":

            width = 0.35

            x = np.arange(len(self.x_axis))

            fig, ax = plt.subplots()
            ax.bar(x - width/2, first_y_axis, width, label= first_label)
            ax.bar(x + width/2, second_y_axis, width, label= second_label)
            ax.set_title(self.title)
            ax.set_xticks(x)
            ax.set_xticklabels(self.x_axis)
            ax.legend()
            
            plt.show()

            
        

if __name__ == '__main__': 
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

    f = open('train_clean.csv', 'w', encoding="utf-8")
    f_test = open('test_clean.csv', 'w', encoding="utf-8")

    bag_of_words = BagOfWords(train_x)
    bag_of_words.create_bag_of_words()

    bag_of_words.add_words_to_bag(test_x)
    bag_of_words.convert_to_list() #bag_of_words.vocab_list

    feature_train, feature_test = generate_feature_set(train_x, test_x, bag_of_words.vocab_list)

    ###Graphing most common words in tweets
    print(train_y.shape)
    print(feature_train.shape)
    real_word_counts  = np.dot(train_y, feature_train)
    fake_word_counts  = np.dot(1-train_y, feature_train)

    most_freq_real = np.argsort(real_word_counts)[::-1]
    most_freq_real = most_freq_real[:10]

    most_freq_fake = np.argsort(fake_word_counts)[::-1]
    most_freq_fake = most_freq_fake[:10]

    most_freq_real_words = [bag_of_words.word_indices[index] for index in most_freq_real]
    most_freq_real_word_counts = [real_word_counts[index] for index in most_freq_real]


    most_freq_fake_words = [bag_of_words.word_indices[index] for index in most_freq_fake]
    most_freq_fake_word_counts = [fake_word_counts[index] for index in most_freq_fake]


    grapher = Grapher(most_freq_real_words, most_freq_real_word_counts, "bar", "Most frequent 10 words in real disaster tweets")

    grapher.plot()

    grapher.change_axises(most_freq_fake_words, most_freq_fake_word_counts)
    grapher.change_title("Most frequent 10 words in fake disaster tweets")

    grapher.plot()

    x_label = ["fire", "disaster", "like", "love", "suicide", "legionnaires"]

    y_real_counts = [real_word_counts[bag_of_words.word_indices.index(word)] for word in x_label]
    y_fake_counts = [fake_word_counts[bag_of_words.word_indices.index(word)] for word in x_label]

    grapher.change_title("Word Comparison for Real and Fake Disaster Tweets")
    grapher.change_plot_type("grouped_bar")
    grapher.change_x_axis(x_label)
    grapher.plot(first_y_axis=y_real_counts, second_y_axis=y_fake_counts,
                 first_label="Real Disaster Tweets", second_label="Fake Disaster Tweets")


    ###creating the classifier using scikit-learn library, splitting data as train and validation, and getting a score based on this
    naive_bayes_model = MultinomialNB(alpha=1.0, fit_prior=True)
    x_train, x_validation, y_train, y_validation = train_test_split( feature_train, train_y, test_size = 0.2)
    naive_bayes_model.fit(x_train, y_train)
    validation_score = naive_bayes_model.score(x_validation, y_validation)
    ###writes test set results into a file
    test_file_name = "predictions_for_test_file.csv"
    predict_for_test_set( feature_test, test_file_name, naive_bayes_model)
    ###prints validation score onto console
    print( validation_score)
    ###prints confusion matrix onto console
    print( confusion_matrix( y_validation, naive_bayes_model.predict_proba(x_validation).argmax(axis=1)))
    tn, fp, fn, tp = confusion_matrix( y_validation, naive_bayes_model.predict_proba(x_validation).argmax(axis=1)).ravel()
    
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision: ", pre)
    print("Recall: ", recall)
    print("F-measure with beta=1: ", 2*pre*recall/(pre + recall))
    print("TPR: ", tp / (tp + fn))
    print("FPR: ", fp / (fp + tn))

    from sklearn.feature_selection import SelectPercentile
    x_train_25percent = SelectPercentile( percentile=25).fit_transform(feature_train, train_y)
    
    naive_bayes_model_new = MultinomialNB(alpha=1.0, fit_prior=True)
    x_train_new, x_validation_new, y_train_new, y_validation_new = train_test_split( x_train_25percent, train_y, test_size = 0.2)
    naive_bayes_model_new.fit(x_train_new, y_train_new)
    validation_score_new = naive_bayes_model_new.score(x_validation_new, y_validation_new)
    print("New: ", validation_score_new)

    x_train_10percent = SelectPercentile( ).fit_transform(feature_train, train_y)
    
    naive_bayes_model_new = MultinomialNB(alpha=1.0, fit_prior=True)
    x_train_new, x_validation_new, y_train_new, y_validation_new = train_test_split( x_train_10percent, train_y, test_size = 0.2)
    naive_bayes_model_new.fit(x_train_new, y_train_new)
    validation_score_new = naive_bayes_model_new.score(x_validation_new, y_validation_new)
    print("New: ", validation_score_new)

    x_train_50percent = SelectPercentile( percentile=50).fit_transform(feature_train, train_y)
    
    naive_bayes_model_new = MultinomialNB(alpha=1.0, fit_prior=True)
    x_train_new, x_validation_new, y_train_new, y_validation_new = train_test_split( x_train_50percent, train_y, test_size = 0.2)
    naive_bayes_model_new.fit(x_train_new, y_train_new)
    validation_score_new = naive_bayes_model_new.score(x_validation_new, y_validation_new)
    print("New: ", validation_score_new)
    
    
    
    
    
    
    
    plot_confusion_matrix( naive_bayes_model, x_validation, y_validation) ###bu pek güzel göstermiyor değerleri

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
