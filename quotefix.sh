#!/bin/sh
cat trump_tweet_archive.csv | sed "s/\"/'/g" > trump_tweet_archive2.csv  
