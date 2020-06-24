import pandas as pd 
import numpy as np
import pickle
from itertools import islice
import subprocess
def window(seq, n=3):
    "Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def addStartEnd(tweet):
    return " ".join(("STARTOFTWEET1","STARTOFTWEET0",tweet,"ENDOFTWEET0"))
if __name__ == "__main__":

    bashCommand = "./quotefix.sh"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    del output,error
    df = pd.read_csv("trump_tweet_archive2.csv")
    df = df[df.is_retweet==False]

    with open("trump_tweet_archive.pickle","wb") as f:
        pickle.dump(df,f)

    with open("most_recent_id.pickle","wb") as f:
        pickle.dump(str(int(df.id_str.iloc[[0]])),f)

    with open("twitter_id_trump.pickle","wb") as f:
        pickle.dump("25073877",f)

    df["text"] = df["text"].apply(addStartEnd)

    conctweets=(" ".join(np.array(df["text"]))).split()

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


    #df[["state1","state2","state3"]] = df[["state1","state2","state3"]].apply(str)
    #df["probabilities"]= df["probabilities"].apply(float)


    with open("transition_probabilities_df.pickle","wb") as f:
        pickle.dump(df,f)

    arr = np.array(df[["state1","state2","state3","probabilities"]])

    with open("transition_probabilities.pickle","wb") as f:
        pickle.dump(arr,f)