import pandas as pd
import numpy as np
# from google.cloud import translate
from googletrans import Translator
import nltk
nltk.download('wordnet')
nltk.download('wordnet')
nltk.download('popular')
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from textblob import TextBlob
# from PyDictionary import PyDictionary
# dictionary=PyDictionary()
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import re
import configparser

config = configparser.ConfigParser()
config.read('src/config.ini')
CONFIG = config

## environment variables
INPUT_FILEPATH = config['project_configuration']['input_filepath']

## dictionaries of aspects and entities
#Aspects-target Dictionary: (Factors motivating plant based choices)

target_aspects_dict = {
   'health': ['health', 'healthy', 'unhealthy','illness','mortality','aging', 'ill-being', 'unhealthy','healthcare', 'fitness', 'weight loss',
              'weightloss','workout', 'Bodybuilding',' fit', 'nutrition', 'selfcare', 'Mentalhealth', 'Mental health'],
   'wellness': ['wellness','unwellness', 'well-being', 'wellbeing','satisfaction', 'satified' 'happiness', 'sadness','happy','sad',
                'dissatisfaction','dissatisfied','contentment','discontentment','content','discontent','stability'],
   'price': ['price', 'inexpensiveness', 'expensivness', 'cost', 'expensive', 'economic', 'value', 'assessment','income', 'cheap',
            'financial'],
   'animal welfare': ['animal welfare', 'Animalwelfare','Animal Rights', 'AnimalRights','animal cruelty','animalcruelty','animal liberation',
                      'animalliberation', 'cruelty free', 'crueltyfree', 'speceism','animal rescue','animals'],
   'environmental sustainability': ['environmental','environment','sustainability', 'ecology','pollution', 'Sustainable', 'ecofriendly',
                                    'climatechange','climate change','climate crisis','climatecrisis', 'unsustainable','global warming',
                                    'globalwarming'],
   'deforestation': ['deforestation', 'wildlife', 'life', 'extinction', 'animals','rainforests'],
   'water': ['ocean','sea', 'waters','water'],
   'taste': ['taste','tasty', 'savour', 'smack', 'verboseness', 'delicious','flavour','aroma','bitter','smoky','sour'],
   'Values': ['ethical', 'moral', 'Ethics', 'morality', 'principles','ideals'],
   'Investment': ['Nasdaq', 'bynd','invest','Investment', 'stocks', 'wallstreet']}
    # 'general': [when there is an entity and no aspect the assign general],


# Target-entities Dictionary (Categorized per research approach)
'''
'Brands, products and organizations: 
Plant Based Abstractions
'''

targets_entities_dict = {
    'Impossible brand': ['Impossible Foods', 'ImpossibleFoods', 'Impossible Meat', 'ImpossibleMeat',
                         'Impossible Burger', 'ImpossibleBurger',
                         'Impossible Whopper', 'ImpossibleWhopper', 'Impossible Sausage', 'ImpossibleSausage',
                         'Impossible pork', 'Impossiblepork',
                         'Plant Based whoper', 'PlantBasedwhoper', 'Impossible breakfast', 'Impossiblebreakfast'],
    'Beyond brand': ['Beyond Meat', 'BeyondMeat', 'Beyond Burger', 'BeyondBurger', 'Beyond Beef', 'BeyondBeef',
                     'Beyond Sausage', 'BeyondSausage',
                     'beyond meat sausage', 'beyondmeatsausage', 'Bynd', 'Beyondchiken', 'Beyond chiken',
                     'beyondfriedchiken', 'beyond fried chiken', 'Go Beyond',
                     'Beyond meatballs', 'Beyondmeatballs', 'Beyond Breakfast', 'BeyondBreakfast', 'Cookout classic',
                     'Cookoutclassic', 'McPlant'],
    'Plant based terms': ['plantbased', 'plant based', 'Vegan', 'Vegetarian', 'Veganism', 'Vegans', 'non-vegans',
                          'Dairyfree', 'Dairy free', 'Dairy-free', 'Meat free', 'Meatless',
                          'Meat alternative', 'fake meat', 'fakemeat', 'clean meat', 'cleanmeat', 'foodtech',
                          'clean meat']}


def translate_text(text=None):
    """
    Transform a one or more tweets within a JSON file to a pandas dataframe with a prior translating function

    :param text:
    :return:
    """
    translator = Translator()
    translated_text = translator.translate(text)
    return translated_text


def from_json_to_pandas(input_filepath=INPUT_FILEPATH, lines=True, topic_to_add=None, language='en', max_rows=None,
                        save=False, output_path=None):
    """ Function that reads a json file with one or more tweets and returns a pandas dataframe. The library should be
    https://github.com/twintproject/twint (twint, not Tweepy, which is the Twitter scraper that we have developed).

    :param output_path:
    :param input_filepath: where the json files containing tweets FROM TWEEPY LIBRARY are stored
    :param lines: if the json file has more than 1 tweet, activate to rue, else False
    :param topic_to_add: a keyword that will be used to create an extra column in the output dataframe as watermark of the origin of these tweets
    :param language: it will filter the dataframe based on the language variable for a preferred language. English by default.
    :param max_rows: if we are going to select non english tweets, or leave the language param as None, which will take a mix of languages, we should
    limit the number of requests as 5 requests/second/user and 200,000 requests/day (Billable limit) with a second limit of 500,000 characters per month for free.
    This means that considering each tweet may reach max limit per Twitter, the maximum number of translations per month is around 1785
    :param save: if True, the function will try to save the filetered dataframe to a csv file
    :return: pandas dataframe
    """
    df = pd.read_json(input_filepath, lines=lines)

    # we filter the dataframe and we keep only the original tweets, this is, we remove the retweeted status ones
    df = df[df['retweet'] == False]

    # we add a watermark topic equal to all observations for each input file to relate that file read to certain topic and origin
    if topic_to_add:
        df['topic'] = topic_to_add

    # we add the language filter given by the language parameter of our function
    if language:
        df = df[df['language'] == language]

    # we save the final dataframe in csv because it has a lower dimension than the input file, therefore we can use the data without
    # using to many resources
    if save:
        df.to_csv(output_path + '.csv')

    return df


def visualize_wordcloud(activate=True, dataframe=None, column='hashtags'):
    """
    Function that plots a wordcloud with the object of distinguish most common words
    :param column: name of the column from where to extract the hashtags
    :param dataframe: original dataframe from previous
    :param activate: visualize a plot of wordclouds from_json_to_pandas function
    :return: frequency distribution of the hashtags
    """
    if activate:
        ## generates a list from all the hashtags in the dataframe
        full_hashtag_list = [x for hashtag_list in dataframe[column].values for x in hashtag_list]

        ## generates a frequency distribution of the hashtags
        fdist = FreqDist(full_hashtag_list)

        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=stopwords,
                              min_font_size=10).generate_from_frequencies(fdist)
        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        # Save image to img folder
        wordcloud.to_file("./first_review.png")
        plt.show()

    return fdist


def hashtag_and_mention_removal(df=None, tweet_column='tweet', colname=None):
    """ This function takes a dataframe returned by the "from_json_to_pandas" function and
    extract hashtags [#], mentions [@] and urls

    :param colname: name of the new column to be added, for tweets without hashtags
    :param tweet_column:
    :param df: original dataframe returned by the from_json_to_pandas() function
    :return:
    """
    tweets = []
    for it, tweet in enumerate(df[tweet_column], 0):
        # hashtag removal that removes ONLY useless hastags that not belong to a certain sentence
        # step 1: removes the pattern, which is [#word][with or without space][#word]
        sentence = re.sub(r'#\w+\s*#\w+', '', tweet)
        # step 2: removes the hash symbol from a hashtag that we consider IS part of a sentence
        sentence = re.sub(r'#', '', sentence)
        # step 3: removes from @whatever, just the @
        sentence = re.sub(r'@', '', sentence)
        # step 4: removes any url in a tweet
        sentence = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            '', sentence)
        tweets.append(sentence)

    df[colname] = tweets

    return df


def tokenize_in_sentences(df=None, id_colname=None, colname=None, subjectivity_threshold=0.5):
    """
    This function takes a dataframe given by the given by hashtag_and_mention_removal() and takes
    the dataframe clean tweets column and creates N observations per tweet, being N the number of
    sentences that such tweet has.

    :param subjectivity_threshold:
    :param df: dataframe returned by previous function hashtag_and_mention_removal()
    :param id_colname: unique id per tweet
    :param colname: name of the column without hashtags, mentions and urls (given by hashtag_and_mention_removal() func)
    :return:
    """
    result_ls = []
    for it1, (idx, id, tweet) in enumerate(zip(df.index, df.id, df[colname])):
        sentences = nltk.sent_tokenize(str(tweet))
        for it2, sentence in enumerate(sentences):
            result_ls.append({'idx': idx, 'id': id, 'it': it2, 'sentence': sentence})

    results_df = pd.DataFrame(result_ls)
    dfx = pd.merge(left=results_df, right=df, how='left', on='id')

    ## add a column with the subjectivity scores for each sentence and filter by equal or more than a rate of 0.5
    # Applying subjectivity scores to the tweets (only sentences df)
    dfx['Subjectivity'] = dfx.sentence.apply(lambda x: TextBlob(str(x)).sentiment[1])
    dfx = dfx[dfx['Subjectivity'] >= subjectivity_threshold]
    dfx.drop(columns=['Subjectivity'], inplace=True)

    return dfx


def aspect_or_entity_extraction(dataframe, dictionary, colname='sentence', transform_to_df=False,
                                df_columns=None, merge=False, join='left', drop_aspect_nas=False):
    """
    Extract all aspects and creates N observations per aspect per sentence

    :param drop_aspect_nas:
    :param join:
    :param merge:
    :param df_columns:
    :param transform_to_df:
    :param colname: where the aspects are (sentence column)
    :param dataframe:
    :param dictionary:
    :return: a pandas.DataFrame object
    """
    global df
    asp_aspterm_sent, final = [], []
    for it, ls in enumerate(list(dictionary.values())):
        final += ls
    for sentence in dataframe[colname]:
        for aspect_term in final:
            for key, value in dictionary.items():
                for word in word_tokenize(sentence):
                    if word.lower() == aspect_term.lower() and str(aspect_term).lower() in \
                            str(sentence).lower() and str(aspect_term).lower() in str(value).lower():
                        asp_aspterm_sent.append((key, aspect_term, sentence))

    asp_aspterm_sent = set(asp_aspterm_sent)
    asp_aspterm = asp_aspterm_sent

    if transform_to_df:
      asp_aspterm_df = pd.DataFrame(asp_aspterm, columns=df_columns)
      df = asp_aspterm_df

    if merge:
        df = pd.merge(df, dataframe, on='sentence', how=join)

    ## substitute None from aspect_term variable to 'general'
    aspect_term_ls = []
    for col in df_columns[1]:
        if (col is np.nan) or (col not in target_aspects_dict.values()) or (col not in target_aspects_dict.values()):
            aspect_term_ls.append('general')
        else:
            aspect_term_ls.append(col)

    df[df_columns[1]] = aspect_term_ls

    ## drop rows that contain nas in aspect
    if drop_aspect_nas:
        df = df[df_columns[1] == 'general']

    return df


def dataframe_to_model(dataframe=None, filter_by='Aspect Category', agg_by='sentence',
                       column_tobe_agg='target'): #, columns_to_drop=['Aspect Category']
    """
    Function that receives a json object with a pandas dataframe structure.
    :param dataframe:
    :param column_tobe_agg:
    :param agg_by:
    :param filter_by: defaults to 'Aspect Category'but we can filter by the column we prefer (this selected column
    will be the groupby column where to make the count)
    :return: features and one-hot encoded label
    """
    df = dataframe[dataframe[filter_by].isnull()].reset_index(drop=True)
    #df = df.drop(columns=columns_to_drop)

    ## filter one :: removes sentences whose entity's frequency falls below a given quantile (IQR)
    quantile_filter = df.groupby(['Aspect Category'])['Aspect Category'].count().to_frame()['Aspect Category'].quantile(0.25)
    mask = df[['Aspect Category']].value_counts() > quantile_filter
    targets_list = [index for index, value in zip(mask.index, mask.values) if value]
    model_data = []
    for sent, te in zip(df.iloc[:, 0], df.iloc[:, 1]):
        if te in targets_list:
            model_data.append((sent, te))
    quantiles_df = pd.DataFrame(model_data).iloc[:, 1].to_frame().value_counts().to_frame()
    targets_list = [index[0] for index, value, in zip(quantiles_df.index, quantiles_df.values)]
    model_data = []
    for sent, te in zip(df.iloc[:, 0], df.iloc[:, 1]):
        if te in targets_list:
            model_data.append((sent, te))

    model_df = pd.DataFrame(model_data)

    ## filter two :: join target-entities grouping by sentences
    model_df.columns = ['sentence', 'target']
    model_df = model_df.drop_duplicates()
    model_df = model_df.groupby([agg_by]).agg({column_tobe_agg: list})
    model_df['sentence'] = model_df.index
    model_df.index = np.arange(model_df.shape[0])

    target_str = [str(item) for item in model_df.target]
    model_df['target_str'] = target_str
    model_df[['target_str']].value_counts()


    ## filter three :: take a sample of size as the minimum frequency


    return model_df


def features_targets_splitter(dataframe):
    """

    :param dataframe:
    :return:
    """
    pass


def add_linguistic_features_to_df(df=None, mode='POS', from_colname=None, new_colname=None, save=False,
                                  saving_format=None, saving_filepath=None):
    """
    This functions inputs a pandas.DataFrame object, executes POS or NER, adds such column to the input dataframe and
    finally, saves or not the output dataframe.

    :param df: input daraframe (it should be pandas.DataFrame object)
    :param mode: either NER or POS
    :param from_colname: the column in which we are going to iterate to extract NER or POS
    :param new_colname: the new column that the function will create and add to the original dataframe with the NER or POS data
    :param save: saves to disk (True/False)
    :param saving_format: json or csv
    :param saving_filepath: os (operating system) filepath where to save the new dataframe
    :return: new dataframe
    """
    # load libraries
    import nltk
    import spacy
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import sent_tokenize
    from nltk import pos_tag
    import re

    # if not os.path.exists()
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    spacy.load('en_core_web_sm')

    # create new variable and adds the new variable to the input dataframe
    if mode is 'POS':
        df[new_colname] = [nltk.pos_tag(word_tokenize(tweet)) for tweet in df[from_colname]]

    if mode is 'NER':
        ner = spacy.load('en_core_web_sm')
        df[new_colname] = [[(ent.text, ent.label_) for ent in ner(tweet).ents] for tweet in df[from_colname]]

    # saves the dataframe into disk
    if saving_format is 'json':
        df.to_json(eval('r') + str(saving_filepath))

    if saving_format is 'csv':
        df.to_csv(saving_filepath)

    return df








