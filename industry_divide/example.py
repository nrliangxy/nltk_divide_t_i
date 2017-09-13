import pandas as pd
import nltk
from nltk.corpus import stopwords
def getTrainData(path):
    df = pd.read_csv(path)
    tken = [word for word in nltk.word_tokenize(df['des'][0])]
    tags = [i[0] for i in nltk.pos_tag(tken) if (("NN" in i[1]) or ("J" in i[1]) or ("RB" in i[1]))]
    filtered_0 = [word.lower() for word in tags if word not in stopwords.words("english")]
    filtered_1 = [word for word in filtered_0 if not any(char.isdigit() for char in word)]
    filtered = [nltk.stem.SnowballStemmer("english").stem(word) for word in filtered_1]
    dict_1 = dict([(i, True) for i in filtered])
    print(filtered)
    print(dict_1)
path = '/home/lxy/Downloads/category/category05.csv'
getTrainData(path)