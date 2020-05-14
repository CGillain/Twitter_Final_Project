import tweepy
import sys
import jsonpickle


#Twitter credentials for the app
consumer_key='M9tJNHCHBAjevlZQQ8t2VKEFc'
consumer_secret='fB9CEGFfCFtsSEtCjBScoFupfanYBSQQgFAjln6aEcFEdLzmDj'
#access_token='1256134844339433473-lwLzbNes91osLE6Xi6AVSjF4aHTnqr'
#access_token_secret='xbzHiWTTL3LEhpo9C0kbe8YvmRiiMBh0GuJX5jmCkcMRy'



# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# Getting Data


searchQuery = ['McDonald', 'McFlurry' ]  # this is what we're searching for
maxTweets = 50000 # Some arbitrary large number
tweetsPerQry = 100  # this is the max the API permits
fName = 'Data_mcdonalds.txt' # We'll store the tweets in a text file.


# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                f.write(jsonpickle.encode(tweet._json, unpicklable=False) +'\n')
                #f.write(json.dumps(tweet) + '\n')
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
