from nltk.classify import NaiveBayesClassifier
import nltk
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet_lemmatizer=WordNetLemmatizer()
def preprocess(sentence):
    words = [word for word in nltk.word_tokenize(sentence)]
    word_list = [wordnet_lemmatizer.lemmatize(word) for word in words]
    #have not pos_tags
    filtered_words = [word.lower() for word in word_list if word not in stopwords.words('english')]
    return {word:True for word in filtered_words}
# tech_data = []
# for line in open('/home/lxy/Documents/train_t_i/tech.json'):
#     tech_data.append([preprocess(json.loads(line)['content']), 'tech'])
# investment_data = []
# for line in open('/home/lxy/Documents/train_t_i/investment.json'):
#     investment_data.append([preprocess(json.loads(line)['content']), 'investment'])

def train_process(path,category):
    data = []
    for line in open(path):
        data.append([preprocess(json.loads(line)['content']), category])
    return data
tech_path = '/home/lxy/Documents/train_t_i/tech.json'
investment_path = '/home/lxy/Documents/train_t_i/investment.json'
train_data = train_process(tech_path,'tech') + train_process(investment_path,'investment')
#train_data = investment_data + tech_data
#testing_data = pos_data[4000:] + neg_data[4000:]
model = NaiveBayesClassifier.train(train_data)
print(model.classify(preprocess('Wave Money , a joint venture between mobile operator Telenor and Yoma Bank, is looking to expand its digital payment services in Myanmar. Telenor and Yoma Bank formed the mobile financial services firm in 2016, by taking a 51 per cent and 49 per cent stake in it respectively. After having spread its agents’ network in 255 out of the 330 townships in Myanmar, Wave Money would further expand into untouched markets in the country, possibly by 2019. “Counting into 2019 that (digital payments) will be definitely something that we will look at, expanding online e-commerce space as well,” Brad Jones, chief executive officer of Wave Money said. The firm claimed to have capitalised on its current network of about 400,000 customers and agents to test other financial services options to further expand presence and fight competition. In the past year, Wave Money largely focussed on money transfer to help people transact in off-banking hours and for the unbanked population. Telecom operator Ooredoo is also poised to roll out its mobile money service named ‘ M-Pitesan ’ soon while local operator Myanma Post and Telecommunications (MPT) is also planning to launch MPT Mobile Money within this year. Some of the services currently being provided by Wave Money include payment for airtime top ups, money transfer and salary disbursements. It has also partnered with organisations such as food delivery platform Food2U , Mobile Ledgends , and ticket booking portal Currently, Wave Money is directly linked to only Yoma Bank for payments. To reach out to a broader audience, the firm is also talking to other banks for partnerships. A grant agreement was signed between Wave Money and the United Nations Capital Development Fund (UNCDF) in January this year for jointly developing an app that will eventually be a mobile game teaching people the basics of financial inclusion.')))