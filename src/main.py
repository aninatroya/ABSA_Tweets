''' add GPU vs CPU option '''
from sys import path
import os
import sys
# sys.path.append(os.path.join(sys.path[0], 'src'))
sys.path.append(os.getcwd() + '//' + 'src')
sys.path.append(os.getcwd())

from load import SListener
from preprocess import from_json_to_pandas, hashtag_and_mention_removal, tokenize_in_sentences
from preprocess import aspect_or_entity_extraction
from preprocess import target_aspects_dict, targets_entities_dict
from preprocess import dataframe_to_model, add_linguistic_features_to_df
from train import train_recurrent_nn, train_bert_nn, get_model, train_semi_bert_nn
from evaluate import custom_classification_report, plot_confusion_matrix
import time
import pandas as pd
import numpy as np
import json
import re
from transformers import GPT2Model

## enironment variables
DOWNLOAD_TWEETS = False
# OUTPUT_FILEPATH =  'data/_streamer_20210322-202719.json' # 'data/tweets.json'
OUTPUT_FILEPATH = 'G:/nlp/Data__BeyondMeat.json' # 'data/tweets.json'


## pipeline
def preprocess_pipeline():
    """

    :return:
    """
    ## step 1: download tweets to a folder in the server (my computer)
    if DOWNLOAD_TWEETS:
        sl = SListener(output_filpath=OUTPUT_FILEPATH)
        sl.execute() ## tested :: ok

    ## step 2: read json with tweets and transform to pandas dataframe
    df = from_json_to_pandas(lines=True, topic_to_add=None, language='en',
                             max_rows=None, save=False, output_path=None) ## tested :: ok

    ## step 3: clean previous dataframe and removes hashtags (#) and mentions (@)
    df = hashtag_and_mention_removal(dataframe=df, tweet_column='tweet', colname='tweet_clean')

    ## step 4: takes previous dataframe (clean) and creates a new dataframe with N sentences per tweet (as observations)
    df = tokenize_in_sentences(dataframe=df, colname='tweet_clean')

    ## step 6: extract all entities and creates N observations per entity per sentence
    df = aspect_or_entity_extraction(dataframe=df, dictionary=targets_entities_dict,
                                     colname='sentence', transform_to_df=True,
                                     df_columns=['Entity category', 'Target-entity', 'sentence'], merge=True)

    ## step 5: extract all aspects and creates N observations per aspect per sentence
    df = aspect_or_entity_extraction(dataframe=df, dictionary=target_aspects_dict,
                                     colname='sentence', transform_to_df=True,
                                     df_columns=['Aspect Category', 'Target-aspect', 'sentence'],
                                     merge=True, join='right')

    ## step 6: function that aggregrates sentences with 1+ aspects
    df = dataframe_to_model(dataframe=df, filter_by='Aspect Category', agg_by='sentence', column_tobe_agg='target')

    ## step 7: function that splits between features and targets
    # not necessary (initially) because training functions do this job

    return df


def training_pipeline(filepath='D:/Data final_to_label_df.xlsx', features=None, target=None,
                      dataframe_for_POS_or_NER=None,
                      task='polarity_classification', model_type='rnn', executable_type='train',
                      activate_semisupervised=False, make_evaluation=True, add_pos_tags=False):
    """
    This function ....

    :param add_pos_tags:
    :param make_evaluation:
    :param activate_semisupervised:
    :param executable_type: 'train' for training, 'transfer_learning' fo TL, 'infer' to make predictions
    for new data (for evaluation purposes or general prediction actions) and 'only_model' to just load a previously
    train model for any purpose.
    :param model_type: to train using a 'rnn' or 'bert'. Just these to options in this code version
    :param filepath: filepath where to load the dataframe
    :param task: two possible options: 'polarity_classification' and 'aspect_extraction'
    :param features: if they come from RAM
    :param target:
    :return:
    """

    ## instead of reading from RAM the parameters features and target, we would ready the dataframe from disk
    global prediction_classes

    if add_pos_tags:
        dataframe = add_linguistic_features_to_df(
            df=None, mode='POS', from_colname=None, new_colname=None, save=False,
            saving_format=None, saving_filepath=None
        )
        dataframe.to_excel('/content/drive/MyDrive/ABSA/sentiment_with_POS.xlsx')

    if task == 'polarity_classification' and model_type == 'rnn':
        ## this first model trains to classify by sentiment using classic RNN architecture
        recnn_model, train_features, train_target = train_recurrent_nn(
            filepath=filepath, train_features=None, train_target=None, test_features=None, test_target=None,
            sentence_colname='sentence', target_colname='target', ohe_target=True, test_size=0.04,
            shuffle=True, save_modelweights_filepath=None, tensorboard_logdir='tensorboard_data/tb_RNN_sentiment',
            save_model_filepath='/models/model_RNN_sentiment.h5'
        )
        return recnn_model, train_features, train_target

    if task == 'polarity_classification' and model_type == 'bert' and not activate_semisupervised:
        bert_model, val_inp, val_mask, val_label = train_bert_nn(filepath=filepath, target_colname='polarity',
                                                                 save_model_filepath='models/bert_model_x.h5')

        return bert_model, val_inp, val_mask, val_label

    if executable_type == 'train':
        bert_model = get_model(
            freeze_layers=True, num_classes=4, train_or_infer='train', train_input=None, train_mask=None,
            train_labels=None, infer_data_input=None, infer_data_mask=None,
            transfer_learning_weights='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights.h5',
            save_model_filepath_w='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_in_callbacks',
            tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_c'
        )
        return  bert_model

    if executable_type == 'infer':
        bert_model, prediction_proba, prediction_classes = get_model(
            freeze_layers=True, num_classes=4, train_or_infer='infer', train_input=None, train_mask=None,
            train_labels=None, infer_data_input=None, infer_data_mask=None,
            transfer_learning_weights='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights.h5',
            save_model_filepath_w='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_in_callbacks',
            tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_c'
        )
        return bert_model, prediction_proba, prediction_classes

    if executable_type == 'only_model':
        bert_model = get_model(
            freeze_layers=True, num_classes=4, train_or_infer='only_model', train_input=None, train_mask=None,
            train_labels=None, infer_data_input=None, infer_data_mask=None,
            transfer_learning_weights='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights.h5',
            save_model_filepath_w='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_in_callbacks',
            tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_c'
        )
        return bert_model

    if task == 'polarity_classification' and model_type == 'bert' and activate_semisupervised:
        bert_model, val_inp, val_mask, val_label = train_semi_bert_nn(
            filepath=None, train_features=None, train_target=None, test_features=None, test_target=None,
            sentence_colname='sentence', target_colname='target', ohe_target=True, test_size=0.10,
            shuffle=True, save_modelweights_filepath=None,
            tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_None',
            save_model_filepath='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_None.hdf5'
        )
        return bert_model, val_inp, val_mask, val_label

    if task == 'aspect_extraction' and model_type == 'bert' and not activate_semisupervised:
        ## DISCLAIMR: this dictionary is just for the ABSA use case of PlantBased, so you should rewrite another one
        ## for your use case
        aspect_dict = {
            'a': 'animal welfare',
            'b': 'BeyondMeat',
            'b ': 'BeyondMeat',
            'bm': 'BeyondMeat',
            's': 'environmental sustainability',
            'h': 'health',
            'pb': 'plantbased',
            'pr': 'price',
            'p': 'price',
            't': 'taste',
            'vn': 'Vegan',
            'vs': 'Veganism',
            'w': 'wellness',
            'w*': 'wellness',

        }
        ## this script cleans the data (we should added to a functio for a better performance in further versions)
        truth_labels = pd.read_excel('/content/drive/MyDrive/ABSA/ground_truth_test_aspect_extraction.xlsx')
        regex_labels = pd.read_excel('/content/drive/MyDrive/ABSA/test_aspect_extraction.xlsx')

        truth_clean = []
        for target in truth_labels.target:
            for pair in aspect_dict.items():
                if pair[0] == target:
                    truth_clean.append(pair[1])

        target_clean = []
        for target in regex_labels.target:
            new = re.sub(r'[\[\]]', '', target)
            new = re.findall(r'\w+', new)
            target_clean.append(new[0])
        regex_labels['target'] = target_clean

        ## the target column is the "aspect" of our mission
        bert_model, val_inp, val_mask, val_label = train_bert_nn(filepath=filepath, target_colname='target',
                                                                 save_model_filepath='models/bert_model_aspect.h5')

        if executable_type == 'infer' and make_evaluation:

            predict_class = [np.argmax(vector) for vector in prediction_classes]
            validation_class = [np.argmax(label) for label in val_label]
            report = custom_classification_report(truth_labels=validation_class, predicted_labels=predict_class)
            plot_confusion_matrix(cm=report,
                                  target_names=[], # unique values of the target variable
                                  title='Confusion matrix',
                                  cmap=None,
                                  normalize=True,
                                  filepath='/content/drive/MyDrive/ABSA/plots/confusion_matrix_aspects.jpg')

            return report


if __name__ == '__main__':
    dataframe = preprocess_pipeline()
    ## multiple may have one or more objects that may be splitted as a list
    multiple = training_pipeline(filepath='D:/Data final_to_label_df.xlsx', features=None, target=None,
                                 dataframe_for_POS_or_NER=dataframe,
                                 task='polarity_classification', model_type='rnn', executable_type='train',
                                 activate_semisupervised=False, make_evaluation=True, add_pos_tags=False)


