''' add GPU vs CPU option '''
from sys import path
import os
import sys
# sys.path.append(os.path.join(sys.path[0], 'src'))
sys.path.append(os.getcwd() + '//' + 'src')
sys.path.append(os.getcwd())


from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import sys

from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import sys

import pandas as pd
import numpy as np

from tweepy.streaming import StreamListener
from tweepy import API
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import sys
# from google.cloud import translate
from googletrans import Translator
import nltk
nltk.download('wordnet')
nltk.download('wordnet')
nltk.download('popular')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from nltk.corpus import wordnet
# from PyDictionary import PyDictionary
# dictionary=PyDictionary()
import matplotlib.pyplot as plt


## function that loads data from Twitter
class SListener(StreamListener):
    API_key = '3gdvDVxyGHvV6rxiAALM3ibWR'
    API_secret = '2FcmbIBKEy10wdxLYprNxBtwqlYz9gKoIrFW7VJQrcSYS44aS6'
    Bearer_token = ''
    Access_token = '1313037754436128768-zTXqZfiDXrOguXDji7vYUQnWEscQf1'
    Access_token_secret = 'E4AsropFEwgBiHFBCLU00YjLYC42kgPcLZVnZvmXRp0e5'

    def __init__(self, api = None, fprefix = 'streamer', api_key='3gdvDVxyGHvV6rxiAALM3ibWR',
                 api_secret='2FcmbIBKEy10wdxLYprNxBtwqlYz9gKoIrFW7VJQrcSYS44aS6',
                 btoken=None, atoken='1313037754436128768-zTXqZfiDXrOguXDji7vYUQnWEscQf1',
                 atoken_secret='E4AsropFEwgBiHFBCLU00YjLYC42kgPcLZVnZvmXRp0e5', output_filpath='data/'):

        ## local variables (codes from a given personal Twitter account)
        self.API_key = api_key
        self.API_secret = api_secret
        self.Bearer_token = btoken
        self.Access_token = atoken
        self.Access_token_secret = atoken_secret

        ## executuion variables
        self.api = api or API()
        self.counter = 0
        self.fprefix = fprefix
        self.output  = open(str(output_filpath) + '_%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')

    def on_data(self, data):
        if 'in_reply_to_status' in data:
            self.on_status(data)
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return

    def on_status(self, status):
        self.output.write(status)
        self.counter += 1
        if self.counter >= 20000:
            self.output.close()
            self.output = open('%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0
        return

    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return

    def on_limit(self, track):
        print("WARNING: Limitation notice received, tweets missed: %d" % track)
        return

    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return

    def on_timeout(self):
        print("Timeout, sleeping for 60 seconds...")
        time.sleep(60)
        return

    @classmethod
    def execute(cls, active=True):
        if active:
            auth = OAuthHandler(cls.API_key, cls.API_secret)
            auth.set_access_token(cls.Access_token, cls.Access_token_secret)
            api = API(auth)

            listen = SListener(api)
            stream = Stream(auth, listen)

            sample_or_keyword = 'keyword'
            if sample_or_keyword == 'sample':
                stream.sample()

            if sample_or_keyword == 'keyword':
                # keywords_to_track = ['BeyondBurger']
                keywords_to_track = ['vegan', 'Vegan', 'BeyondBurger', 'BeyondMeat']
                stream.filter(track=keywords_to_track)
