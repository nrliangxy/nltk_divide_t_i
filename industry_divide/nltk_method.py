#from __future__ import absolute_import, division, print_function
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords

def getTrainData(path,num):
	df=pd.read_csv(path)
	trainingset=[]
	for i in df['des']:
		tken=[word for word in nltk.word_tokenize(i)]
		#print ( [i for i in nltk.pos_tag(tken) if not (("NN" in i[1]) or ( "J" in i[1]) or ( "RB" in i[1]) )]    )
		tags= [i[0] for i in nltk.pos_tag(tken) if (("NN" in i[1]) or ( "J" in i[1]) or ( "RB" in i[1])) ]
		#tags=tken
		###filter stop words
		filtered_0=[word.lower() for word in tags if word not in stopwords.words("english")]
		###filter numbers dates
		filtered_1=[word for word in filtered_0 if not any(char.isdigit() for char in word) ]
		###filter dot and "-"
		#filtered_2=[word for word in filtered_1 if (not ("-" in word)) and (not ("," in word)) ]
		###stem words
		filtered=[nltk.stem.SnowballStemmer("english").stem(word) for word in filtered_1]
		dict_1=dict([(i,True) for i in filtered ])
		#dict_1=dict(nltk.FreqDist(filtered))
		#print dict_1
		###only need big categories [01-09],[10-19],[20-39],[40-49],[50-59],[60-69],[70-89],[90-99]
		if num//10 == 0:labl="0Agriculture, Forestry & Fishing"
		elif 15>num>9: labl="1Mining"
		elif 18>num>14: labl="2Construction"
		elif 40>num>19: labl="3Manufacturing"
		elif num//10 == 4: labl="4Transportation, Communications, Electric, Gas & Sanitary Services"
		elif 52>num>49 : labl="5Wholesale Trade"
		elif 60>num>51: labl="6Retail Trade"
		elif 68>num>59: labl="7Finance, Insurance & Real Estate"
		elif 90>num>69: labl="8services"
		elif 99>num>90: labl="9Public Administration"
		trainingset.append((dict_1,labl))
	return trainingset

def dumpModel(model,path):
	f=open(path,"wb")
	modelDump=pickle.dump(model,f)
	f.close()
	return modelDump

def loadModel(path):
	f=open(path,"rb")
	model=pickle.load(f)
	f.close()
	return model

#train43_set=getTrainData("category/category62.csv",62)
#train42_set=getTrainData("category/category74.csv",74)
#train44_set=getTrainData("category/category47.csv",47)
#train45_set=getTrainData("category/category96.csv",96)

#midNum=-10
#train_set=train42_set[:midNum]+train43_set[:midNum]+train44_set[:midNum]+train45_set[:midNum]
#test_set=train42_set[midNum:]+train43_set[midNum:]+train44_set[midNum:]+train45_set[midNum:]
#print (train_set)

train_set=[]
test_set=[]
for i in range(100):
	try:
		train_test_all=getTrainData("/home/lxy/Downloads/category_1/category%.2d.csv"%i,i)
		if len(train_test_all)>10:
			train_set.extend(train_test_all[:-5])
			test_set.extend(train_test_all[-5:])
			print (i,len(train_set),len(test_set))
	except:continue

print ("training_amount",len(train_set))
print (">>>>>>>>>>")
#print (test_set)
print ("testing_amount",len(test_set))
print (">>>>>>>>>>>>>")
classifier_0 = nltk.NaiveBayesClassifier.train(train_set)
dumpModel(classifier_0,"/home/lxy/Downloads/model_all.pickle")
classifier_1=loadModel("/home/lxy/Downloads/model_all.pickle")


test="this company is for farming and plastic"
test0=dict([(i,True) for i in test.split()])
print ('Accuracy: %.4f'%nltk.classify.accuracy(classifier_1, test_set))
print (classifier_1.show_most_informative_features())
#for test_set in test
print (classifier_1.classify(test0))