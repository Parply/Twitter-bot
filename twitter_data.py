import tweepy
import pickle

def getApi():
    keys=pickle.load(open("api_keys.pickle","rb"))


    auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
    auth.set_access_token(keys["access_token_key"], keys["access_token_secret"])

    api = tweepy.API(auth)
    return api


#25073877






