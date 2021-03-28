# bot.py
import os
import pandas as pd
import discord
from dotenv import load_dotenv
from yahoo_fin import stock_info as si
import stockPrediction
import decimal
import yfinance as yf
from datetime import datetime, timedelta

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

client = discord.Client()

userPortfolios = {}

def getPrice(stockName):
    # get the stock price from 3 days ago (for weekend testing)
    data = yf.download(tickers=stockName, period="5d", interval="1m")
    d = datetime.today() - timedelta(days=3)
    newD = d.strftime("%Y-%m-%d %H:%M")+':00-04:00' # force the input string to the right format
    return data.loc[[newD]].values[0][1]   # return the opening value of that particular minute

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

def buy(stockName, amount, user):
    global userPortfolios

    try:
        userPortfolio = userPortfolios[user]
    except:
        userPortfolios[user] = {'cash':100000}
        userPortfolio = userPortfolios[user]

    try:
        numStock = userPortfolio[stockName]
        numStock += amount
    except:
        numStock = amount

    #stockPrice =  si.get_live_price(stockName) # current stock price
    stockPrice = getPrice(stockName) # get data from 3 days ago

    userCash = userPortfolio['cash']
    userCash -= stockPrice*numStock
    userPortfolio['cash'] = round(userCash*10000)/10000

    userPortfolio[stockName] = numStock
    print(userPortfolio)

    userPortfolios[user] = userPortfolio
    return userPortfolio

def sell(stockName, amount, user):
    global userPortfolios

    try:
        userPortfolio = userPortfolios[user]
    except:
        userPortfolios[user] = {'cash':100000}
        userPortfolio = userPortfolios[user]

    try:
        numStock = userPortfolio[stockName]
        numStock -= amount
    except:
        raise Exception('you do not own this stock')

    if(numStock<0):
        raise Exception('you cannot sell more than you have')

    #stockPrice =  si.get_live_price(stockName) # current stock price
    stockPrice = getPrice(stockName) # get data from 3 days ago

    userCash = userPortfolio['cash']
    userCash += stockPrice*amount
    userPortfolio['cash'] = round(userCash*10000)/10000

    userPortfolio[stockName] = numStock
    print(userPortfolio)

    userPortfolios[user] = userPortfolio
    return userPortfolio

def getStandings():
    global userPortfolios

    standings = []

    for user, userPortfolio in userPortfolios.items():
        worth = 0
        for stock, amount in userPortfolio.items():
            if(stock == 'cash'):
                worth += amount
            else:
                #stockPrice =  si.get_live_price(stock) # current stock price
                stockPrice = getPrice(stock) # get data from 3 days ago
                worth += stockPrice*amount
        
        standings.append( (user, worth) )
    
    standings.sort(key = lambda x: x[1], reverse=True) 
    print(standings)
    if(len(standings) < 5):
        numRanks = len(standings)
    else:
        numRanks = 5
    
    outputStr = ""
    for i in range(numRanks):
        outputStr += str(i+1)+". "+ str(standings[i][0]) + " : " + str(round(standings[i][1]*10000)/10000) + "\n"

    return outputStr

def getPortfolio(user):
    global userPortfolios

    try:
        userPortfolio = userPortfolios[user]
    except:
        userPortfolios[user] = {'cash':100000}
        userPortfolio = userPortfolios[user]

    return userPortfolio

def resetPortfolio(user):
    global userPortfolios
    userPortfolios[user] = {'cash':100000}

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    cmd = message.content
    if cmd.startswith('!buy'):
        try:
            messagelist = [x.strip() for x in cmd.split(" ")]
            stockName = messagelist[1].upper()
            userPortfolio  = buy(stockName, int(messagelist[2]), message.author)
            
            # build output string of all user holdings
            outputString = ""
            for stock, amount in userPortfolio.items():
                outputString += stock + ":" + str(amount) + "\n"
                print(outputString)

            await message.channel.send(outputString)
        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    if cmd.startswith('!sell'):
        try:
            messagelist = [x.strip() for x in cmd.split(" ")]
            stockName = messagelist[1].upper()
            userPortfolio  = sell(stockName, int(messagelist[2]), message.author)
            
            # build output string of all user holdings
            outputString = ""
            for stock, amount in userPortfolio.items():
                outputString += stock + ":" + str(amount) + "\n"
                print(outputString)

            await message.channel.send(outputString)
        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    if cmd.startswith('!price'):
        try:
            messagelist = [x.strip() for x in cmd.split(" ")]
            stockName = messagelist[1].upper()
            #stockPrice =  si.get_live_price(stockName) # current stock price
            stockPrice = getPrice(stockName) # get data from 3 days ago
            if(stockName == "GME"):
                await message.channel.send('{:.4f}'.format(stockPrice)+ "\nhttps://giphy.com/gifs/bitcoin-cryptocurrency-diamond-hands-IXWqceoF995QxrBnL0")
            else:
                await message.channel.send('{:.4f}'.format(stockPrice))
        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    if cmd.startswith('!predict'):
        try:
            messagelist = [x.strip() for x in cmd.split(" ")]
            stockPrediction.tomorrow(messagelist[1].upper(), years=3, plot=True)
            with open('test.png', 'rb') as f:
                picture = discord.File(f)
                await message.channel.send(file=picture)
        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    if cmd.startswith("!standing"):
        outputString = getStandings()
        await message.channel.send(outputString)

    if cmd.startswith("!portfolio"):
        try:
            userPortfolio  = getPortfolio(message.author)
            
            # build output string of all user holdings
            outputString = ""
            for stock, amount in userPortfolio.items():
                outputString += stock + " : " + str(amount) + "\n"
                print(outputString)

            await message.channel.send(outputString)
        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    if cmd.startswith("!bankrupt"):
        try:
            resetPortfolio(message.author)
            userPortfolio  = getPortfolio(message.author)
            
            # build output string of all user holdings
            outputString = ""
            for stock, amount in userPortfolio.items():
                outputString += stock + " : " + str(amount) + "\n"
                print(outputString)

            await message.channel.send(outputString)

        except Exception as e:
            if(str(e) == "list index out of range"):
                await message.channel.send("ENCOUNTERED ERROR: not enough params")
            else:
                await message.channel.send("ENCOUNTERED ERROR: " + str(e))

    greeting = 'HOWDY'

    helpInfo = ''' !hello - greet Stocky
    !buy stockname # - buy stock
    !sell stockname # - sell stock
    !price stockname - get price of stock
    !predict stockname - get stock prediction graph
    !standing - output top players (max of 5) and their worth
    !portfolio - output your cash and stock
    !bankrupt - reset portfolio with $100000'''

    if message.content == '!hello':
        response = greeting
        await message.channel.send(response)
    elif message.content == 'raise-exception':
        raise discord.DiscordException

    if message.content == '!help':
        await message.channel.send(helpInfo)
    elif message.content == 'raise-exception':
        raise discord.DiscordException


client.run(TOKEN)