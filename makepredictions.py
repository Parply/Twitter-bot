import pandas as pd
import numpy as np
import pickle
from collections import deque
from twitter_data import *
from make_transition import *
import schedule
import json
import time
import logging

logger = logging.getLogger('twitterpi')
hdlr = logging.FileHandler('./twitterpi.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

def predict(max_words=180):
    tweet = []
    queue = deque(["STARTOFTWEET1","STARTOFTWEET0"])
    logger.debug("Loading transition probabilities")
    with open("transition_probabilities.pickle","rb") as f:
        trans_prob = pickle.load(f)
    logger.debug("Loaded transition probabilities")
    word = ""
    max_words =max_words+1
    
    while (len(tweet) <= max_words) and (word != "ENDOFTWEET0"):
        tweet += [word]
        ind = trans_prob[:,0]==queue[0]
        p = trans_prob[ind]
        ind = p[:,1]==queue[1]
        p = p[ind]
        ind =np.random.choice(p.shape[0],1,p=p[:,3].astype(np.float64))[0]
        word = p[ind,2]
        queue.popleft()
        queue.append(word)
    logger.debug("Generated a tweet")
    tweet = np.array(tweet[1:])
    ind = tweet == "&amp;"
    tweet[ind] = "&"
    tweet = " ".join(tweet[1:])
    logger.debug(f"Tweet: {tweet}")
    return tweet

def tweet():
    logger.debug("PREDICTING A TWEET")
    logger.debug("Loading twitter API")
    api = getApi()
    tweet = ""
    logger.debug("Calling predicted")
    while tweet == "":
        tweet = predict()
    logger.debug("Updating status")
    status = api.update_status(tweet)
    logger.debug(f"Posted a tweet: {status.text}")
    return f"Donald Trump's tweet: {status.text}"


def updateTransProbs():
    logger.debug("UPDATING TRANSITION PROBABILITIES")
    logger.debug("Loading twitter API")
    api = getApi()
    logger.debug("Loading most recent status ID")
    with open("most_recent_id.pickle","rb") as f:
        most_recent_id = pickle.load(f)
    logger.debug("Loading twitter ID")
    with open("twitter_id_trump.pickle","rb") as f:
        twitter_id = pickle.load(f)
    logger.debug("Pulling tweets")
    tweets = api.user_timeline(id=twitter_id,since_id=most_recent_id)
    tweets = np.array([[i.text,i.id_str] for i in tweets if not i.text.startswith("RT @")])    
    
    print(f"Got {tweets.shape[0]} new tweets")
    logger.debug(f"Number of new tweets {tweets.shape[0]}")
    if tweets.shape[0]!=0:

        most_recent_id = str(tweets[:,1].astype(np.int64).max())
        tweets = np.array(list(map(addStartEnd,tweets[:,0])))
        logger.debug("Opening old tweets")
        with open("conc_tweets.pickle","rb") as f:
            conctweets=pickle.load(f)

        logger.debug("Adding the new tweets")
        conctweets=(" ".join(np.array(tweets))).split() + conctweets
        logger.debug("Saving concatinated tweets")
        with open("conc_tweets.pickle","wb") as f:
            pickle.dump(conctweets,f)
        logger.debug("Getting transitions")
        df = pd.DataFrame(window(conctweets), columns=['state1', 'state2','state3'])
        df = df[df.state1!="ENDOFTWEET0"]
        df = df[df.state2!="ENDOFTWEET0"]
        logger.debug("Counting transitions")
        counts = df.groupby(['state1','state2'])['state3'].value_counts()
        df = pd.DataFrame(np.hstack(((np.array(list(counts.index)),np.array(counts)[:,None]))),columns=["state1","state2","state3","counts"])
        df=df.astype({"counts":"int"})


        logger.debug("Calculating probabilities")
        df["id"] = df.groupby(["state1","state2"]).ngroup()
        ids = np.array(df["id"])
        counts=np.array(df["counts"])
        scounts = np.bincount(ids, weights=counts)
        reps=np.array(df.groupby(["state1","state2"]).size())
        scounts=np.repeat(scounts,reps)
        df["scounts"] = scounts
        df["probabilities"] = df["counts"]/df["scounts"]

        df = df.astype({"state1":"str","state2":"str","state3":"str","probabilities":"float"})
        logger.debug("Saving transition dataframe")
        with open("transition_probabilities_df.pickle","wb") as f:
                pickle.dump(df,f)

        arr = np.array(df[["state1","state2","state3","probabilities"]])
        logger.debug("Saving transition array")
        with open("transition_probabilities.pickle","wb") as f:
            pickle.dump(arr,f)
        logger.debug("Saving new most recent tweet ID")
        with open("most_recent_id.pickle","wb") as f:
            pickle.dump(most_recent_id,f)
    return None
updateTransProbs()
tweet()
schedule.every(4).to(8).hours.do(tweet)
schedule.every(12).hours.do(updateTransProbs)
while True:
    schedule.run_pending()
    time.sleep(60)