# <center> ABSA_Tweets </center>
## <center> Aspect-based Sentiment Analysis of Social Media Data with Pre-trained Language Models </center>

## 1. Introduction
### 1. Abstract
There is a great scope in utilizing the increasing content expressed by users on social media platforms such as Twitter. This study explores the application of Aspect-based Sentiment Analysis (ABSA) of tweets to retrieve fine-grained sentiment insights. The Plant based food domain is chosen as an area of focus. To the best of our knowledge this is the first time ABSA task is done for this sector and it is distinct from standard food products because different and controversial aspects arise and opinions are polarized. The choice is relevant because these products can help in meeting the sustainable development goals and improve the welfare of millions of animals. Pre-trained BERT,"Bidirectional Encoder Representations with transformers", is fine-tuned for this task and stands out because it was trained to learn from all the words in the sentence simultaneously using transformers. The aim was to develop methods to be applied on real life cases, therefore lowering the dependency on labeled data and improving performance were the key objectives. This research contributes to existing approaches of ABSA by proposing data processing techniques to adapt social media data for ABSA. The scope of this project presents a new method for the aspect category detection task (ACD) which does not rely on labeled data by using regular expressions (Regex). For aspect the sentiment classification task (ASC) a semi-supervised learning technique is explored. Additionally Part-of-Speech (POS) tags are incorporated into the predictions. The findings show that Regex is a solution to eliminate the dependency on labeled data for ACD. For ASC fine-tuning BERT on a small subset of data was the most accurate method to lower the dependency on aspect level sentiment data.

Th noteboook presents the steps conducting Aspect-Based Sentiment Analysis of social media data starting from retrieving the data and finalizing with a confusion matrix to asses how the diffrent model versions are pefrorme for this task.


## 2. How to use this repo?
### 2.1. Download the source code to your local machine
1. Download the remote repository into your system (PC or server).

  - git clone https://github.com/aninatroya/ABSA_Tweets.git

2. Download necessary dependencies.

  - pip install -r requirements.txt

3. Access the configuration file in src/config.ini and write down your configuration parameters

### 2.2. The "load.py" file
Initially, there is nothing to modify as the configuration parameters come from the config.ini file, where the user should put his/her project configuration parameters.

### 2.3. The "preprocess.py" file
- The configuration parameters are read from the config.ini file directly from the "environment variables" of the file. 
- There are two main dictionaries for the use case "target_aspects_dict" and "targets_entities_dict", **that should be modified directly** by the user, depending on her/his use cases. The targets that appear in the tweets are those that are referred to by the aspect/entity dictionary. Consecutively any comment (tweet) about a given target aspect or target entity, will be labeled with their respective entities. For your use case, you should **take care and modify both dictionaries, as they are related**.
