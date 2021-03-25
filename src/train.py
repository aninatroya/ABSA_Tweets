import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, Flatten, BatchNormalization, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, GPT2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
import tensorflow as tf
import keras
from keras_radam import RAdam
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig, AlbertTokenizer, TFAlbertModel
import tensorflow_addons as tfa
import pickle
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import sentencepiece
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, Flatten, BatchNormalization, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, GPT2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
import tensorflow as tf
import keras
from keras_radam import RAdam
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig, AlbertTokenizer, TFAlbertModel
import pickle
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import sentencepiece
import matplotlib.pyplot as plt

## pandas dataframe configuration parameters
# initial table display config:
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 40)
pd.options.display.float_format = '{:.2f}'.format


def train_recurrent_nn(filepath=None, train_features=None, train_target=None, test_features=None, test_target=None,
                       sentence_colname='sentence', target_colname='target', ohe_target=True, test_size=0.04,
                       shuffle=True, save_modelweights_filepath='models/recurrent_rnn_weights.h5',
                       tensorboard_logdir='tensorbord/tb_rnn', save_model_filepath='models/recurrent_rnn.h5'):
    """
    :param save_modelweights_filepath:
    :param tensorboard_logdir:
    :param save_model_filepath:
    :param shuffle:
    :param test_size:
    :param ohe_target:
    :param target_colname:
    :param sentence_colname:
    :param filepath:
    :param train_features:
    :param train_target:
    :param test_features:
    :param test_target:
    :return:
    """

    ## special variables
    SEED = 1

    ## aspect case :: json file
    if filepath is not None and 'json' in filepath:

        ## tokenization, tokens to numbers based on corpus IDs and padding
        dataframe = pd.read_json(filepath)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])
        X = pad_sequences(sentences_as_numeric, padding='post')
        target = pd.Series(dataframe[target_colname], dtype='category').cat.codes
        if ohe_target:
            target = to_categorical(target)

        ## train test split
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=SEED,
                                                            shuffle=bool(shuffle))
    else:
        X_train, X_test, y_train, y_test = train_features, test_features, train_target, test_target

    ## polarity case :: xlsx file
    if filepath is not None and 'xlsx' in filepath:

        dataframe = pd.read_excel(filepath)
        dataframe.drop(columns=['Unnamed: 0', 'Unnamed: 4'], inplace=True)
        dataframe.columns = ['sentence', 'target', 'polarity']
        dataframe.drop([134], inplace=True)  # we drop nan observations (just one row)

        target_sentiment_dict = {
            'positive': ['p', 'p*'],
            'negative': ['n', 'n*'],
            'neutral': ['r', 'r*'],
            'conflict': ['c', 'c*']  # conflict for sarcasm / double sense / irony
        }

        polarity_full = []
        for pol in dataframe[target_colname]:
            for key, val in target_sentiment_dict.items():
                if pol in val:
                    polarity_full.append(key)

        dataframe['polarity_full'] = polarity_full
        dataframe.polarity = dataframe.polarity_full
        dataframe.drop(columns=['polarity_full'], inplace=True)
        dataframe['syn'] = dataframe.apply(lambda x: x.sentence + ' ' + x.target, axis=1)
        dataframe['sentence'] = dataframe.syn
        dataframe.drop(columns=['target', 'syn'], inplace=True)

        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])  # X (Xtrain, xtest)
        X = pad_sequences(sentences_as_numeric, padding='post')  # this means to fill will zeros at the end (due 'post')

        target = pd.Series(dataframe['polarity'], dtype='category').cat.codes
        target = to_categorical(target)

        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.04, random_state=8374,
                                                            shuffle=True)

    ## reshaping
    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], 1))

    model = Sequential(name='RecurrentNN')
    model.add(layer=LSTM(units=1000, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(layer=LSTM(units=800, return_sequences=True, activation='relu'))
    model.add(layer=GRU(units=800, return_sequences=True, activation='relu'))
    model.add(layer=LSTM(units=600, return_sequences=True, activation='relu'))
    ## spatial-temporal feature extraction
    model.add(layer=tf.keras.layers.Conv1D(filters=400, kernel_size=3, padding='same', activation='relu'))
    model.add(layer=MaxPooling1D(pool_size=2, padding='same')) # dimensionality reduction
    model.add(layer=GRU(units=600, return_sequences=False, activation='relu'))
    model.add(layer=Dropout(rate=0.15))
    model.add(layer=Dense(units=10, activation='softmax'))
    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_modelweights_filepath, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=tensorboard_logdir),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)]

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.Accuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) #, epsilon=1e-08)
    optimal_optimizer_radam = tfa.optimizers.RectifiedAdam(learning_rate=2e-4)
    optimizer_beta = RAdam(learning_rate=2e-4, epsilon=1e-08, name='radam')

    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=12, verbose=1, validation_split=0.1,
                        callbacks=callbacks)
    model.save(str(save_model_filepath))

    return model, X_test, y_test


# model, X_test, y_test = train_recurrent_nn('D:/data_for_data_aspect_extraction.json')


def train_bert_nn(filepath=None, train_features=None, train_target=None, test_features=None, test_target=None,
                  sentence_colname='sentence', target_colname='target', ohe_target=True, test_size=0.10,
                  shuffle=True, save_modelweights_filepath=None,
                  tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment',
                  save_model_filepath='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights.h5'):
    """
    :param filepath:
    :param train_features:
    :param train_target:
    :param test_features:
    :param test_target:
    :param sentence_colname:
    :param target_colname:
    :param ohe_target:
    :param test_size:
    :param shuffle:
    :param save_modelweights_filepath:
    :param tensorboard_logdir:
    :param save_model_filepath:
    :return:
    """
    ## special variables
    tokenizer, dataframe, target = 0, 0, 0
    SEED = 1

    ## aspect case :: json file
    if filepath is not None and 'json' in filepath:

        ## tokenization, tokens to numbers based on corpus IDs and padding
        dataframe = pd.read_json(filepath)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])
        X = pad_sequences(sentences_as_numeric, padding='post')
        target = pd.Series(dataframe[target_colname], dtype='category').cat.codes
        if ohe_target:
            target = to_categorical(target)

        ## train test split
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=SEED,## It seems that this train test split happens if the file is .json
                                                            shuffle=bool(shuffle))
    else:
        X_train, X_test, y_train, y_test = train_features, test_features, train_target, test_target

    ## polarity case :: xlsx file
    if filepath is not None and 'xlsx' in filepath:

        dataframe = pd.read_excel(filepath)
        # dataframe.drop(columns=['Unnamed: 0', 'Unnamed: 4'], inplace=True)
        dataframe.drop(columns=['Unnamed: 0'], inplace=True)
        dataframe.columns = ['sentence', 'target', 'polarity']
        dataframe.drop([134], inplace=True)  # we drop nan observations (just one row)

        target_sentiment_dict = {
            'positive': ['p', 'p*'],
            'negative': ['n', 'n*'],
            'neutral': ['r', 'r*'],
            'conflict': ['c', 'c*']  # conflict for sarcasm / double sense / irony
        }

        polarity_full = []
        for pol in dataframe[target_colname]:
            for key, val in target_sentiment_dict.items():
                if pol in val:
                    polarity_full.append(key)

        dataframe['polarity_full'] = polarity_full
        dataframe.polarity = dataframe.polarity_full
        dataframe.drop(columns=['polarity_full'], inplace=True)
        dataframe['syn'] = dataframe.apply(lambda x: x.sentence + ' ' + x.target, axis=1)
        dataframe['sentence'] = dataframe.syn
        dataframe.drop(columns=['target', 'syn'], inplace=True)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])  # X (Xtrain, xtest)
        X = pad_sequences(sentences_as_numeric, padding='post')  # this means to fill will zeros at the end (due 'post')

        target = pd.Series(dataframe['polarity'], dtype='category').cat.codes
        target = to_categorical(target)

        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=8374, ## It seems that this train test split happens if the file is .xlsx
                                                            shuffle=True)

    ## reshaping
    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], 1))

    ## load tokenizer and model
    num_classes = len(dataframe.polarity.unique())
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # based on unique words  30,522 words + BPE
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    ## add emojis to the tokenizer
    emotions_and_people = ['😀','😃','😄','😁','😆','😅','🤣','😇','😉','😊','🙂','🙃','☺','😋','😌','😍','🥰','😘','😗','😙','😚','🤪','🤪','😜','😝','😛','🤑','😎','🤓',
                    '🧐','🤠','🥳','🤗','🤡','😏','😶','😐','😑','😒','🙄','🤨','🤔','🤫','🤭','🤥','😳','😞','😟','😠','😡','🤬','😔','😕','🙁','🤯','😲','🥴','😵',
                    '🤩','😭','😓','🤤','😪','😥','😢','😧','😦','😯','😰','😨','😱','😮','😤','😩','😫','😖','😣','🥺','😬','☹','🤐','😷','🤕','🤒','🤮','🤢','🤧',
                    '🥵','🥶','😴','💤','😈','👿','👹','👺','💩','👻','💀','☠','👽','🤖','🎃','😺','😸','😹','😻','😼','☝','👇','👆','👉','👈','👌','🤟','🤘','✌',
                    '🤞','🤜','🤛','✊','👊','👎','👍','🤝','🙏','👏','🙌','🤲','👐','😾','😿','🙀','😽','✋','🤚','🖐','🖖','👋','🤙','💪','🖕','💆‍♂️','💆','💆‍♀️','🤦‍♂️',
                    '🤦','🤦‍♀️','👩‍❤️‍👨','👩‍❤️‍👩','💑','👨‍❤️‍👨','💟','💝','💘','💖','💗','💓','💞','💕','❣','💔','🖤','💜','💙','💚','💛','🧡','❤','👨‍❤️‍💋‍👨','💏','👩‍❤️‍💋‍👩','👩‍❤️‍💋‍👨','🐶','🐱','🐭',
                    '🐹','🐰','🐻','🧸','🐼','🐨','🐯','🦁','🐮','🐷','🐽','🐸','🐵','🙈','🙉','🙊','🐒','🦍','🐔','🐧','🐦','🐤','🐣','🐍','🐢','🦠','🦟','🦂','🕸','🕷',
                    '🦗','🐜','🐞','🐌','🦋','🐛','🐝','🦄','🦘','🦌','🦒','🦓','🐴','🐗','🦝','🐺','🦊','🐥','🦎','🐙','🦑','🦞','🦀','🦐','🐠','🐟','🐡','🐬','🦈','🐳',
                    '🐋','🐊','🐆','🐅','🐃','🐂','🐄','🐪','🐫','🦙','🐘','🦏','🦛','🐐','🦔','🦡','🐿','🐁','🐀','🐇','🐈','🐩','🐕','🦜','🦚','🦉','🦢','🦆','🦅','🕊',
                    '🦃','🐓','🦇','🐖','🐎','🐑','🐏','🐾','🌲','🌳','🌴','🌸','🌞','☄','✨','🌟','💫','⭐','🌜','🌛','🌝','🌚','🌙','🌔','🌓','🌒','🌑','🌘','🌗','🌖',
                    '🌕','🌏','🌍','🌎','🐚','🌰','🍄','💐','☀','🌤','⛅','🌥','☁','⛈','🌩','⚡','🔥','💥','❄','🌨','☃','⛄','🌬','💨','🌪','🌫','🌈','☔','💧','💦',
                    '🌊','🍏','🍎','🍐','🍊','🍌','🍋','🍉','🍇','🍓','🍈','🍒','🍑','🥭','🍍','🥥','🥝','🍅','🥑','🍆','🌶','🥒','🥬','🥦','🌽','🥕','🌮','🥪','🍝','🍕',
                    '🌭','🍟','🍔','🥓','🍳','🥚','🍤','🥩','🍖','🍗','🧀','🥞','🥯','🥨','🥖','🥐','🍞','🍯','🥜','🍠','🥔','🥗','🌯','🥙','🍜','🥘','🍲','🥫','🧂','🍥',
                    '🍣','🍱','🍛','🍙','🍚','🍘','🥟','🍢','🍡','🍨','🍧','🍦','🍰','🎂','🧁','🥧','🍮','🥄','🍶','🍾','🍹','🍸','🥃','🥂','🍷','🍻','🍺','🥛','🥤','🍼',
                    '🍵','🥣','☕','🥮','🥠','🍪','🍩','🍿','🍫','🍬','🍭','🍴','🍽','🥢','🥡','🧘‍♂️','🧘','🧘‍♀️','🎖','🏅','🥇','🥈','🎯','♟','🧩','🎲','🎻','👮‍♀️','👮','👮‍♂️',
                    '👨‍🔧','👩‍🌾','🧑‍🌾','👨‍🌾','👩‍🍳','🧑‍🍳','👨‍🍳','🧘‍♂️','🧘','🧘‍♀️','🎖','🏅','🥇','🥈','🎯','♟','🧩','🎲','🎻','👮‍♀️','👮','👮‍♂️','👨‍🔧','👩‍🌾','🧑‍🌾','👨‍🌾','👩‍🍳','🧑‍🍳','👨‍🍳'
                    '💋','💄','🎈','🎏','🎁','🧧','🎀','🎊','🎉','🚿','🛁','🛀','🌡','💉','💊','🧨','💣','🔪','🗡','⚔','🛡','🚬','⚰','⚱','🏺','💰','💷','💶','💴','💵',
                    '💸','❗','❕','❓','‼','⁉','💯','♻','🆖','🆗','🆙','🆒','🆕','🆓','🚮','🔝','🛒']

    for emoji in emotions_and_people:
        bert_tokenizer.add_tokens(str(emoji))

    ## encoding the labels
    input_ids, attention_masks = [], []
    for sent in dataframe[sentence_colname]:
        bert_inp = bert_tokenizer.encode_plus(text=sent, add_special_tokens=True, max_length=X_train.shape[1],
                                              return_attention_mask=True, pad_to_max_length=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    ## conversion of all encodings to numpy arrays
    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(target)
    assert len(input_ids) == len(attention_masks) == len(labels)

    ## load and save data
    ## saving and loading the data into pickle files
    pickle_inp_path = '/content/drive/MyDrive/ABSA/data/bert_inp_sentiment_pos.pkl'
    pickle_mask_path = '/content/drive/MyDrive/ABSA/data/bert_mask_sentiment_pos.pkl'
    pickle_label_path = '/content/drive/MyDrive/ABSA/data/bert_label_sentiment_pos.pkl'

    pickle.dump((input_ids), open(pickle_inp_path, 'wb'))
    pickle.dump((attention_masks), open(pickle_mask_path, 'wb'))
    pickle.dump((labels), open(pickle_label_path, 'wb'))

    input_ids = pickle.load(open(pickle_inp_path, 'rb'))
    attention_masks = pickle.load(open(pickle_mask_path, 'rb'))
    labels = pickle.load(open(pickle_label_path, 'rb'))

    ## splitting into train and validation set
    train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(input_ids,
                                                                                        labels,
                                                                                        attention_masks,
                                                                                        test_size=test_size)

    ## setting up the callbacks, loss, metric and the optimizer
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_model_filepath, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=tensorboard_logdir), EarlyStopping(monitor='val_loss', patience=6)]

    print('\nBert Model', bert_model.summary())

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # to CategoricalCrossentropy [ohe vector]
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')  # to CategoricalCrossentropy [ohe vector]
    metric = tf.keras.metrics.CategoricalAccuracy('categorical_accuracy')  # to CategoricalCrossentropy [ohe vector]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)

    ## freeze layers
    for it, layer in enumerate(bert_model.layers, 1):
        if it == len(bert_model.layers):
            break
        else:
            layer.trainable = False

    ## drop classification layer from previous architecture and add our custom classification layer
    # bert_model._layers.pop(-1)
    # bert_model._layers.pop(-1)
    # bert_model._layers = bert_model._layers.__add__([Dense(10, activation='softmax')])

    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    bert_model.summary()
    # history = bert_model.fit(x=[train_inp, train_mask], y=train_label, batch_size=32, epochs=4,
    #                          validation_data=([val_inp, val_mask], val_label), callbacks=callbacks)

    history = bert_model.fit(x=[train_inp, train_mask], y=train_label, batch_size=32, epochs=1,
                             validation_split=test_size, callbacks=callbacks)

    ## save model after training
    if save_model_filepath is not None:
        bert_model.save_weights(save_model_filepath) #, save_format='tf')

    return bert_model, val_inp, val_mask, val_label


def get_model(freeze_layers=True, num_classes=4, train_or_infer='train', train_input=None, train_mask=None,
              train_labels=None, infer_data_input=None, infer_data_mask=None,
              transfer_learning_weights='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights.h5',
              save_model_filepath_w='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_in_callbacks',
              tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_c'):
    """
    :param freeze_layers:
    :param num_classes:
    :param train_or_infer:
    :param train_input:
    :param train_mask:
    :param train_labels:
    :param infer_data_input:
    :param infer_data_mask:
    :param transfer_learning_weights:
    :param save_model_filepath_w:
    :param tensorboard_logdir:
    :return:
    """
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    ## setting up the callbacks, loss, metric and the optimizer
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_model_filepath_w, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)]

    print('\nBert Model', bert_model.summary())

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # to CategoricalCrossentropy [ohe vector]
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')  # to CategoricalCrossentropy [ohe vector]
    metric = tf.keras.metrics.CategoricalAccuracy('categorical_accuracy')  # to CategoricalCrossentropy [ohe vector]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-08)

    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    bert_model.summary()

    ## load previous model
    bert_model.load_weights(str(transfer_learning_weights))

    ## freeze layers
    if freeze_layers:
        for it, layer in enumerate(bert_model.layers, 1):
            if it == len(bert_model.layers):
                break
            else:
                layer.trainable = False

    if 'train' in train_or_infer:
        history = bert_model.fit(x=[train_input, train_mask], y=train_labels, batch_size=16, epochs=5,
                                 validation_split=0.10, callbacks=callbacks)
        return bert_model

    if 'infer' in train_or_infer:
        prediction_proba = bert_model.predict([infer_data_input, infer_data_mask])
        prediction_classes = np.array([np.argmax(pred_proba) for pred_proba in prediction_proba[0]])
        return bert_model, prediction_proba, prediction_classes
    if 'only_model' in train_or_infer:
        return bert_model


def train_semi_bert_nn(filepath=None, train_features=None, train_target=None, test_features=None, test_target=None,
                       sentence_colname='sentence', target_colname='target', ohe_target=True, test_size=0.10,
                       shuffle=True, save_modelweights_filepath=None,
                       tensorboard_logdir='/content/drive/MyDrive/ABSA/tensorboard_data/tb_bert_sentiment_None',
                       save_model_filepath='/content/drive/MyDrive/ABSA/models/bert_model_sentiment_weights_None.hdf5'):
    """
    :param filepath:
    :param train_features:
    :param train_target:
    :param test_features:
    :param test_target:
    :param sentence_colname:
    :param target_colname:
    :param ohe_target:
    :param test_size:
    :param shuffle:
    :param save_modelweights_filepath:
    :param tensorboard_logdir:
    :param save_model_filepath:
    :return:
    """
    ## special variables
    tokenizer, dataframe, target = 0, 0, 0
    SEED = 1

    ## aspect case :: json file
    if filepath is not None and 'json' in filepath:

        ## tokenization, tokens to numbers based on corpus IDs and padding
        dataframe = pd.read_json(filepath)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])
        X = pad_sequences(sentences_as_numeric, padding='post')
        target = pd.Series(dataframe[target_colname], dtype='category').cat.codes
        if ohe_target:
            target = to_categorical(target)

        ## train test split
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=SEED,
                                                            ## It seems that this train test split happens if the file is .json
                                                            shuffle=bool(shuffle))
    else:
        X_train, X_test, y_train, y_test = train_features, test_features, train_target, test_target

    ## polarity case :: xlsx file
    if filepath is not None and 'xlsx' in filepath:

        dataframe = pd.read_excel(filepath)
        # dataframe.drop(columns=['Unnamed: 0', 'Unnamed: 4'], inplace=True)
        dataframe.drop(columns=['Unnamed: 0'], inplace=True)
        dataframe.columns = ['sentence', 'target', 'polarity']
        dataframe.drop([134], inplace=True)  # we drop nan observations (just one row)
        dataframe.dropna(inplace=True)

        target_sentiment_dict = {
            'positive': ['p', 'p*'],
            'negative': ['n', 'n*'],
            'neutral': ['r', 'r*'],
            'conflict': ['c', 'c*']  # conflict for sarcasm / double sense / irony
        }

        polarity_full = []
        for pol in dataframe[target_colname]:
            for key, val in target_sentiment_dict.items():
                if pol in val:
                    polarity_full.append(key)
                # else:
                #     polarity_full.append(key)

        dataframe['polarity_full'] = polarity_full
        dataframe.polarity = dataframe.polarity_full
        dataframe.drop(columns=['polarity_full'], inplace=True)
        dataframe['syn'] = dataframe.apply(lambda x: x.sentence + ' ' + x.target, axis=1)
        dataframe['sentence'] = dataframe.syn
        dataframe.drop(columns=['target', 'syn'], inplace=True)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataframe[sentence_colname])
        sentences_as_numeric = tokenizer.texts_to_sequences(dataframe[sentence_colname])  # X (Xtrain, xtest)
        X = pad_sequences(sentences_as_numeric, padding='post')  # this means to fill will zeros at the end (due 'post')

        target = pd.Series(dataframe['polarity'], dtype='category').cat.codes
        target = to_categorical(target)

        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=8374,
                                                            ## It seems that this train test split happens if the file is .xlsx
                                                            shuffle=True)

    ## reshaping
    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], 1))

    ## load tokenizer and model
    num_classes = len(dataframe.polarity.unique())
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # based on unique words  30,522 words + BPE
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    ## add emojis to the tokenizer
    emotions_and_people = ['😀', '😃', '😄', '😁', '😆', '😅', '🤣', '😇', '😉', '😊', '🙂', '🙃', '☺', '😋', '😌',
                           '😍', '🥰', '😘', '😗', '😙', '😚', '🤪', '🤪', '😜', '😝', '😛', '🤑', '😎', '🤓',
                           '🧐', '🤠', '🥳', '🤗', '🤡', '😏', '😶', '😐', '😑', '😒', '🙄', '🤨', '🤔', '🤫', '🤭',
                           '🤥', '😳', '😞', '😟', '😠', '😡', '🤬', '😔', '😕', '🙁', '🤯', '😲', '🥴', '😵',
                           '🤩', '😭', '😓', '🤤', '😪', '😥', '😢', '😧', '😦', '😯', '😰', '😨', '😱', '😮', '😤',
                           '😩', '😫', '😖', '😣', '🥺', '😬', '☹', '🤐', '😷', '🤕', '🤒', '🤮', '🤢', '🤧',
                           '🥵', '🥶', '😴', '💤', '😈', '👿', '👹', '👺', '💩', '👻', '💀', '☠', '👽', '🤖', '🎃',
                           '😺', '😸', '😹', '😻', '😼', '☝', '👇', '👆', '👉', '👈', '👌', '🤟', '🤘', '✌',
                           '🤞', '🤜', '🤛', '✊', '👊', '👎', '👍', '🤝', '🙏', '👏', '🙌', '🤲', '👐', '😾', '😿',
                           '🙀', '😽', '✋', '🤚', '🖐', '🖖', '👋', '🤙', '💪', '🖕', '💆‍♂️', '💆', '💆‍♀️', '🤦‍♂️',
                           '🤦', '🤦‍♀️', '👩‍❤️‍👨', '👩‍❤️‍👩', '💑', '👨‍❤️‍👨', '💟', '💝', '💘', '💖', '💗', '💓',
                           '💞', '💕', '❣', '💔', '🖤', '💜', '💙', '💚', '💛', '🧡', '❤', '👨‍❤️‍💋‍👨', '💏',
                           '👩‍❤️‍💋‍👩', '👩‍❤️‍💋‍👨', '🐶', '🐱', '🐭',
                           '🐹', '🐰', '🐻', '🧸', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐽', '🐸', '🐵', '🙈', '🙉',
                           '🙊', '🐒', '🦍', '🐔', '🐧', '🐦', '🐤', '🐣', '🐍', '🐢', '🦠', '🦟', '🦂', '🕸', '🕷',
                           '🦗', '🐜', '🐞', '🐌', '🦋', '🐛', '🐝', '🦄', '🦘', '🦌', '🦒', '🦓', '🐴', '🐗', '🦝',
                           '🐺', '🦊', '🐥', '🦎', '🐙', '🦑', '🦞', '🦀', '🦐', '🐠', '🐟', '🐡', '🐬', '🦈', '🐳',
                           '🐋', '🐊', '🐆', '🐅', '🐃', '🐂', '🐄', '🐪', '🐫', '🦙', '🐘', '🦏', '🦛', '🐐', '🦔',
                           '🦡', '🐿', '🐁', '🐀', '🐇', '🐈', '🐩', '🐕', '🦜', '🦚', '🦉', '🦢', '🦆', '🦅', '🕊',
                           '🦃', '🐓', '🦇', '🐖', '🐎', '🐑', '🐏', '🐾', '🌲', '🌳', '🌴', '🌸', '🌞', '☄', '✨', '🌟',
                           '💫', '⭐', '🌜', '🌛', '🌝', '🌚', '🌙', '🌔', '🌓', '🌒', '🌑', '🌘', '🌗', '🌖',
                           '🌕', '🌏', '🌍', '🌎', '🐚', '🌰', '🍄', '💐', '☀', '🌤', '⛅', '🌥', '☁', '⛈', '🌩', '⚡',
                           '🔥', '💥', '❄', '🌨', '☃', '⛄', '🌬', '💨', '🌪', '🌫', '🌈', '☔', '💧', '💦',
                           '🌊', '🍏', '🍎', '🍐', '🍊', '🍌', '🍋', '🍉', '🍇', '🍓', '🍈', '🍒', '🍑', '🥭', '🍍',
                           '🥥', '🥝', '🍅', '🥑', '🍆', '🌶', '🥒', '🥬', '🥦', '🌽', '🥕', '🌮', '🥪', '🍝', '🍕',
                           '🌭', '🍟', '🍔', '🥓', '🍳', '🥚', '🍤', '🥩', '🍖', '🍗', '🧀', '🥞', '🥯', '🥨', '🥖',
                           '🥐', '🍞', '🍯', '🥜', '🍠', '🥔', '🥗', '🌯', '🥙', '🍜', '🥘', '🍲', '🥫', '🧂', '🍥',
                           '🍣', '🍱', '🍛', '🍙', '🍚', '🍘', '🥟', '🍢', '🍡', '🍨', '🍧', '🍦', '🍰', '🎂', '🧁',
                           '🥧', '🍮', '🥄', '🍶', '🍾', '🍹', '🍸', '🥃', '🥂', '🍷', '🍻', '🍺', '🥛', '🥤', '🍼',
                           '🍵', '🥣', '☕', '🥮', '🥠', '🍪', '🍩', '🍿', '🍫', '🍬', '🍭', '🍴', '🍽', '🥢', '🥡',
                           '🧘‍♂️', '🧘', '🧘‍♀️', '🎖', '🏅', '🥇', '🥈', '🎯', '♟', '🧩', '🎲', '🎻', '👮‍♀️', '👮',
                           '👮‍♂️',
                           '👨‍🔧', '👩‍🌾', '🧑‍🌾', '👨‍🌾', '👩‍🍳', '🧑‍🍳', '👨‍🍳', '🧘‍♂️', '🧘', '🧘‍♀️', '🎖',
                           '🏅', '🥇', '🥈', '🎯', '♟', '🧩', '🎲', '🎻', '👮‍♀️', '👮', '👮‍♂️', '👨‍🔧', '👩‍🌾',
                           '🧑‍🌾', '👨‍🌾', '👩‍🍳', '🧑‍🍳', '👨‍🍳'
                                                               '💋', '💄', '🎈', '🎏', '🎁', '🧧', '🎀', '🎊', '🎉',
                           '🚿', '🛁', '🛀', '🌡', '💉', '💊', '🧨', '💣', '🔪', '🗡', '⚔', '🛡', '🚬', '⚰', '⚱', '🏺',
                           '💰', '💷', '💶', '💴', '💵',
                           '💸', '❗', '❕', '❓', '‼', '⁉', '💯', '♻', '🆖', '🆗', '🆙', '🆒', '🆕', '🆓', '🚮', '🔝',
                           '🛒']

    for emoji in emotions_and_people:
        bert_tokenizer.add_tokens(str(emoji))

    ## encoding the labels
    input_ids, attention_masks = [], []
    for sent in dataframe[sentence_colname]:
        bert_inp = bert_tokenizer.encode_plus(text=sent, add_special_tokens=True, max_length=X_train.shape[1],
                                              return_attention_mask=True, pad_to_max_length=True)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    ## conversion of all encodings to numpy arrays
    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(target)
    assert len(input_ids) == len(attention_masks) == len(labels)

    ## load and save data
    ## saving and loading the data into pickle files
    pickle_inp_path = '/content/drive/MyDrive/ABSA/data/bert_inp_sentiment_semi.pkl'
    pickle_mask_path = '/content/drive/MyDrive/ABSA/data/bert_mask_sentiment_semi.pkl'
    pickle_label_path = '/content/drive/MyDrive/ABSA/data/bert_label_sentiment_semi.pkl'

    pickle.dump((input_ids), open(pickle_inp_path, 'wb'))
    pickle.dump((attention_masks), open(pickle_mask_path, 'wb'))
    pickle.dump((labels), open(pickle_label_path, 'wb'))

    input_ids = pickle.load(open(pickle_inp_path, 'rb'))
    attention_masks = pickle.load(open(pickle_mask_path, 'rb'))
    labels = pickle.load(open(pickle_label_path, 'rb'))

    ## splitting into train and validation set
    train_inp, val_inp, train_label, val_label, train_mask, val_mask = train_test_split(input_ids,
                                                                                        labels,
                                                                                        attention_masks,
                                                                                        test_size=test_size)

    ## setting up the callbacks, loss, metric and the optimizer
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_model_filepath, save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=tensorboard_logdir), EarlyStopping(monitor='val_loss', patience=6)]

    print('\nBert Model', bert_model.summary())

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # to CategoricalCrossentropy [ohe vector]
    # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')  # to CategoricalCrossentropy [ohe vector]
    metric = tf.keras.metrics.CategoricalAccuracy('categorical_accuracy')  # to CategoricalCrossentropy [ohe vector]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-08)

    ## freeze layers
    for it, layer in enumerate(bert_model.layers, 1):
        if it == len(bert_model.layers):
            break
        else:
            layer.trainable = False

    ## drop classification layer from previous architecture and add our custom classification layer
    # bert_model._layers.pop(-1)
    # bert_model._layers.pop(-1)
    # bert_model._layers = bert_model._layers.__add__([Dense(10, activation='softmax')])

    bert_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    bert_model.summary()
    # history = bert_model.fit(x=[train_inp, train_mask], y=train_label, batch_size=32, epochs=4,
    #                          validation_data=([val_inp, val_mask], val_label), callbacks=callbacks)

    history = bert_model.fit(x=[train_inp, train_mask], y=train_label, batch_size=32, epochs=1,
                             validation_split=test_size, callbacks=callbacks)

    ## save model after training
    if save_model_filepath is not None:
        bert_model.save_weights(save_model_filepath)  # , save_format='tf')

    return bert_model, val_inp, val_mask, val_label
