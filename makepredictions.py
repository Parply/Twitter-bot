import pandas as pd
import numpy as np
import pickle
from collections import deque
from twitter_data import *
from make_transition import *
import schedule
import json
import time

def predict(max_words=180):
    tweet = []
    queue = deque(["STARTOFTWEET1","STARTOFTWEET0"])
    
    with open("transition_probabilities.pickle","rb") as f:
        trans_prob = pickle.load(f)
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
    tweet = np.array(tweet[1:])
    ind = tweet == "&amp;"
    tweet[ind] = "&"
    tweet = " ".join(tweet[1:])
    
    return tweet

def tweet():
    api = getApi()
    tweet = ""
    while tweet == "":
        tweet = predict()
    status = api.update_status(tweet)
    return f"Donald Trump's tweet: {status.text}"


def updateTransProbs():
    api = getApi()

    with open("most_recent_id.pickle","rb") as f:
        most_recent_id = pickle.load(f)

    with open("twitter_id_trump.pickle","rb") as f:
        twitter_id = pickle.load(f)
    
    tweets = api.user_timeline(id=twitter_id,since_id=most_recent_id)
    tweets = np.array([[i.text,i.id_str] for i in tweets if not i.text.startswith("RT @")])    
    
    print(f"Got {tweets.shape[0]} new tweets")
    if tweets.shape[0]!=0:
        most_recent_id = str(tweets[:,1].astype(int).max())
        tweets = np.array(list(map(addStartEnd,tweets[:,0])))

        with open("conc_tweets.pickle","rb") as f:
            conctweets=pickle.load(f)


        conctweets=(" ".join(np.array(tweets))).split() + conctweets

        with open("conc_tweets.pickle","wb") as f:
            pickle.dump(conctweets,f)

        df = pd.DataFrame(window(conctweets), columns=['state1', 'state2','state3'])
        df = df[df.state1!="ENDOFTWEET0"]
        df = df[df.state2!="ENDOFTWEET0"]
        counts = df.groupby(['state1','state2'])['state3'].value_counts()
        df = pd.DataFrame(np.hstack(((np.array(list(counts.index)),np.array(counts)[:,None]))),columns=["state1","state2","state3","counts"])
        df=df.astype({"counts":"int"})


        df["id"] = df.groupby(["state1","state2"]).ngroup()
        ids = np.array(df["id"])
        counts=np.array(df["counts"])
        scounts = np.bincount(ids, weights=counts)
        reps=np.array(df.groupby(["state1","state2"]).size())
        scounts=np.repeat(scounts,reps)
        df["scounts"] = scounts
        df["probabilities"] = df["counts"]/df["scounts"]

        df = df.astype({"state1":"str","state2":"str","state3":"str","probabilities":"float"})
        with open("transition_probabilities_df.pickle","wb") as f:
                pickle.dump(df,f)

        arr = np.array(df[["state1","state2","state3","probabilities"]])

        with open("transition_probabilities.pickle","wb") as f:
            pickle.dump(arr,f)
        with open("most_recent_id.pickle","wb") as f:
            pickle.dump(most_recent_id,f)
    return df
if __name__=="__main__":
    schedule.every(4).to(8).hours.do(tweet)
    schedule.every(12).hours.do(updateTransProbs)
    while True:
        schedule.run_pending()
        time.sleep(60)