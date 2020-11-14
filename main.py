import time
import pandas as pd
import argparse
import sqlite3
import requests
import csv
import folium


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
    info, key = extractArgs(run)
    str = analyseCsvFile(info)
    df = createDataFrame()
    getCoordinates(df, key)
    mapCoordinates()

def cmdArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--input_transactions_file", required=True)
    parser.add_argument("-k", "--coordinates_key", required=True)

    args = parser.parse_args()
    return args

def extractArgs(args):
    transactionsFile = open(args.input_transactions_file, encoding='utf-8-sig')
    transactionInfo = transactionsFile.readlines()

    key = args.coordinates_key
    return transactionInfo, key

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

def createDataFrame():
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

    for x in range(0, 3): #change to len(df) later
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

def catagorizeTransactions():

def transactionStatistics():


def transactionCharts():

runProgramNoDB()
