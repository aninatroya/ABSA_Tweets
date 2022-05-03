''' add GPU vs CPU option '''
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
from typing import List, Optional
# from google.cloud import translate
import nltk
nltk.download('wordnet')
nltk.download('wordnet')
nltk.download('popular')
import configparser

config = configparser.ConfigParser()
config.read('src/config.ini')
CONFIG = config

# from PyDictionary import PyDictionary
# dictionary=PyDictionary()

## function that loads data from Twitter
class SListener(StreamListener):
    """
    Class to download tweets directly from the Twitter API
    """
    API_key = config['project_configuration']['api_key']
    API_secret = config['project_configuration']['api_secret']
    Bearer_token = ''
    Access_token = config['project_configuration']['atoken']
    Access_token_secret = config['project_configuration']['atoken_secret']

    def __init__(self, api = None, fprefix = 'streamer', api_key='',
                 api_secret='',
                 btoken=None, atoken='',
                 atoken_secret='', output_filpath=config['project_configuration']['output_filepath']):

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
        self.output = open(str(output_filpath) + '_%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')

    def on_data(self, data):
        """
        Function to parse the data received from the Twitter API.

        :param data: not usage
        :return: None
        """
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
            return None

    def on_status(self, status):
        """
        Function to parse the data received from the Twitter API.
        Parameters
        ----------
        status

        Returns
        -------

        """
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
    def execute(cls, active=True, sample_or_keyword='keyword', keywords_to_track: List = None):
        """
        This method, if executed, starts downloading tweets with the
        :param sample_or_keyword: if 'keyword', the keywords_to_track argument should be passed with a list with
        those words that if appear in a tweet, it will be downloaded. If 'sample', a sample will be downloaded
        without limits
        :param keywords_to_track: list of words, if one of those is within a tweet, it will be automatically downloaded
        :param active: if True, the program will be executed
        :return: None
        """
        if active:
            auth = OAuthHandler(cls.API_key, cls.API_secret)
            auth.set_access_token(cls.Access_token, cls.Access_token_secret)
            api = API(auth)

            listen = SListener(api)
            stream = Stream(auth, listen)

            if sample_or_keyword == 'sample':
                stream.sample()

            if sample_or_keyword == 'keyword':
                stream.filter(track=keywords_to_track)

        return None

