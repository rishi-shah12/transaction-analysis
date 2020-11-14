import time
import pandas as pd
import argparse
import sqlite3
import requests
import csv
import folium
import bs4
import urllib
import sys
import yfinance as yf
import nltk
import sklearn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.classify.scikitlearn import SklearnClassifier
from bs4 import BeautifulSoup

def runProgram():
    run = cmdArguments()
    info, key = extractArgs(run)
    str = analyseCsvFile(info)
    connect = createDataBase(str)
    df = createDataFrame()
    writeToDataBase(df, connect)
    getCoordinates(df, key)
    mapCoordinates()

def runProgramNoDB():
    run = cmdArguments()
    infoFile, info, key, companies = extractArgs(run)
    str = analyseCsvFile(info)
    df = createDataFrame(infoFile)
    getCoordinates(df, key)
    mapCoordinates()

def cmdArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--input_transactions_file", required=True)
    parser.add_argument("-k", "--coordinates_key", required=True)
    parser.add_argument("-c", "--companies_dataset", required=True)

    args = parser.parse_args()
    return args

def extractArgs(args):
    transactionsFile = open(args.input_transactions_file, encoding='utf-8-sig')
    transactionInfo = transactionsFile.readlines()

    key = args.coordinates_key

    companiesFile = open(args.companies_dataset, encoding='utf-8-sig')
    companies = transactionsFile.readlines()
    return transactionsFile, transactionInfo, key, companiesFile

def analyseCsvFile(transactionsFile):
    headingsWithCommas = transactionsFile[0]
    headingsWithCommas = headingsWithCommas.strip()
    headingsWithoutCommas = headingsWithCommas.split(',')
    return headingsWithoutCommas


def createDataBase(headings):
    connect = sqlite3.connect('transactions.db')
    db = connect.cursor()
    db.execute('CREATE TABLE transactions (Date text, Time time, City text, Province text, Country text, Merchant text, Price real)')
    return connect

def createDataFrame(infoFile):
    data = pd.read_csv('transactions.csv')
    df = pd.DataFrame(data)
    return df

def writeToDataBase(dataFrame, connect):
    dataFrame.to_sql('transactions', connect, if_exists='replace', index=False)

def getCoordinates(df, key):
    URL = "https://us1.locationiq.com/v1/search.php?key=" + key
    with open('locations.csv', 'w', newline='') as f_output:
        titles = ['name','lat','lon']
        location_output = csv.writer(f_output, delimiter=',')
        location_output.writerow(titles)

    for x in range(0, len(df)): #change to len(df) later
        print(x)

        street = (df["Address"][x]).replace(" ","")
        city = df["City"][x]
        country = df["Country"][x]
        URL = URL + "&street=" + street + "&city=" +city + "&country=" + country + "&format=json"

        request = requests.get(url=URL)
        data = request.json()
        latitude = data[0]['lat']
        longitude = data[0]['lon']
        print(latitude, longitude)

        info = [city, latitude, longitude]
        with open('locations.csv', 'a', newline='') as file_output:
            info_output = csv.writer(file_output, delimiter=',')
            info_output.writerow(info)
        time.sleep(0.5)

def mapCoordinates():
    tooltip = "Click for detailed info"
    m = folium.Map(location=[43.64877, -79.38171])
    coordinatesFile = open('locations.csv')
    coordinatesFileLines = coordinatesFile.readlines()

    for line in coordinatesFileLines:
        currentLine = line.split(',')
        if currentLine[0] == 'name':
            continue
        else:
            folium.Marker([currentLine[1], currentLine[2]], popup='<i>'+currentLine[0]+'</i>', tooltip=tooltip).add_to(m)

    m.save('transaction_locations.html')

def loadDataSet(companyDataSet):
    df = pd.read_csv(companyDataSet, header=None)
    df = df.astype(str)
    return df

def preprocessDataSet(dataSet):
    encoder = LabelEncoder()
    classes = dataSet[1]
    Y = encoder.fit_transform(classes)
    descriptions = dataSet[0]
    return descriptions, Y

def replaceText(descriptions):
    processed = descriptions.str.replace(r'€|\$','moneysymb')
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    processed = processed.str.replace(r'\s+', ' ')
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    return processed

def modifyText(replacedText):
    modifiedText = replacedText.str.lower()
    modifiedText = modifiedText.str.strip()
    stopWords = set(stopwords.words('english'))
    modifiedText = modifiedText.apply(lambda x: ' '.join(term for term in x.split() if term not in stopWords))
    ps = nltk.PorterStemmer()
    modifiedText = modifiedText.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
    return modifiedText

def tokenizeWords(modifiedText):
    allWords = []
    for description in modifiedText:
        words = word_tokenize(description)
        for w in words:
            allWords.append(w)
    allWords = nltk.FreqDist(allWords)

    wordFeatures = []
    for i in range(520):
        wordFeatures.append(allWords.most_common()[i][0])

    return wordFeatures

def findFeatures(description, wordFeatures):
    words = word_tokenize(description)
    features = {}
    for word in wordFeatures:
        features[word] = (word in words)
    return features

def findFeaturesAll(processedText, Y, wordFeatures):
    descriptions = list(zip(processedText, Y))

    seed = 1
    np.random.seed = seed
    np.random.shuffle(descriptions)

    featureSets = [(findFeatures(text, wordFeatures), label) for (text, label) in descriptions]

    return featureSets, seed

def trainingTestingDataset(featureSets, seed):
    training, testing = model_selection.train_test_split(featureSets, test_size=0.2, random_state=seed)
    return training, testing

def modelsToTrain(training, testing):
    names = ['K Nearest Neighbours', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'SGDClassifier', 'SVM Linear']

    rfc = RandomForestClassifier(n)
    nltk_model = SklearnClassifier(rfc)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
    print("accuracy: " + str(accuracy) + "%")

    # classifiers = [
    #     KNeighborsClassifier(),
    #     DecisionTreeClassifier(),
    #     RandomForestClassifier(),
    #     LogisticRegression(),
    #     SGDClassifier(max_iter=100),
    #     SVC(kernel='linear')
    # ]
    #
    # models = list(zip(names, classifiers))
    #
    # for name, model in models:
    #     nltk_model = SklearnClassifier(rfc)
    #     nltk_model.train(training)
    #     accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
    #     print("accuracy: " + name + " " + str(accuracy) + "%")


#def catagorizeTransactions():



#def transactionStatistics():


#def transactionCharts():


run = cmdArguments()
infoFile, info, key, companies = extractArgs(run)
data = loadDataSet(companies)
desc, y = preprocessDataSet(data)
replacedText = replaceText(desc)
processedText = modifyText(replacedText)
wordfeat = tokenizeWords(processedText)
featureSets, seed = findFeaturesAll(processedText, y, wordfeat)

train, test = trainingTestingDataset(featureSets, seed)
modelsToTrain(train, test)










