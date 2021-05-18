from numpy.core.fromnumeric import put
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open('./src/core/intents.json') as file:
    data = json.load(file)

try:
    with open("./src/core/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"]) 

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
	model.load("./src/core/model.tflearn")
except:
	model.fit(training, output, n_epoch=1000000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(str, words):
    bag = [0 for _ in range(len(words))]

    str_words = nltk.word_tokenize(str)
    str_words = [stemmer.stem(word.lower()) for word in str_words]

    for s in str_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Entrer votre texte ici (taper exit pour quitter)")
    while True:
        inp = input("Vous: ")
        if inp.lower() == "exit":
            break

        res = model.predict([bag_of_words(inp, words)])
        res_index = numpy.argmax(res)
        tag = labels[res_index]
        
        for id in data["intents"]:
            if id['tag'] == tag:
                responses = id['responses']
        
        print(responses)

def chat2(message):
	res = model.predict([bag_of_words(message, words)])
	res_index = numpy.argmax(res)
	tag = labels[res_index]

	for id in data["intents"]:
		if id['tag'] == tag:
			responses = id['responses']

	return str(responses)
#chat()