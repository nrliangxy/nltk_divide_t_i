from nltk.classify import NaiveBayesClassifier
import nltk
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet_lemmatizer=WordNetLemmatizer()
def preprocess(sentence):
    words = [word for word in nltk.word_tokenize(sentence)]
    word_list = [wordnet_lemmatizer.lemmatize(word) for word in words]
    filtered_words = [word.lower() for word in word_list if word not in stopwords.words('english')]
    return {word:True for word in filtered_words}

def train_process(path,category):
    data = []
    for line in pd.read_json(path,lines=True)['content']:
        data.append([preprocess(line), category])
    return data
tech_path = '/home/lxy/tech.json'
investment_path = '/home/lxy/investment.json'
train_data = train_process(tech_path,'tech')[:4800] + train_process(investment_path,'investment')[:4800]
model = NaiveBayesClassifier.train(train_data)
with open('/home/lxy/Documents/nbs_5000.pickle','wb') as fw:
    pickle.dump(model,fw)
#print(model.classify(preprocess('Chinese co-working office space URWork is moving a step further to catch up with its U.S. peer WeWork, at least in valuation. The company announced today the completion of 400 million RM')))