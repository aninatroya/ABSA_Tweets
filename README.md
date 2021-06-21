# <center> ABSA_Tweets </center>
## <center> Aspect-based Sentiment Analysis of Social Media Data with Pre-trained Language Models </center>

## 1. Introduction
### 1. Abstract
In customer centric businesses there is a great scope in utilizing the insights expressed by the target groups on social media platformssuch as Twitter. This study explores the application of Aspect-based Sentiment Analysis (ABSA) of tweets to retrieve fine-grainedinsights about the different aspects present and the respective senti-ments expressed about them. For accurate results this methodologyuses pre-trained deep learning language models. In this domain theBERT architecture stands out because it was trained to learn fromall the words in the sentence simultaneously by using transformers.With regards to the architecture of ABSA with BERT this researchcontributes to existing approaches by using a semi-supervised learn-ing technique and incorporating POS tags into the predictions. Asan area of focus the Plant-based food domain is chosen. To the bestof our knowledge this is the first time ABSA task is done for thissector. In this analysis plant-based is distinct from standard foodproducts because different and controversial aspects arise in thetext and opinions are polarized. The choice is relevant because suchproducts can help in meeting the sustainable development goals andimprove the welfare of millions of animals. Moreover the purposeof this research is to develop a practical model that can be usedon real time streaming of data. It is expected that the applicationpre-trained BERT with semi-supervised learning will lead to thebest performance possible while minimizing the dependency onlabeled data.

The following noteboook presents a stepwise approach to conducting Aspect-Based Sentiment Analysis of social media data starting from the initial step of retrieving the data and finalizing with a confusion matrix to asses how the finetuned model is performing on the data.

### 2. 
fsdfs

### 3. 
dasdsa

## 2. How to use this repo
### 2.1. Download the source code to your local machine
1. Download remote repository into your system (PC or server)

  - git clone https://github.com/aninatroya/ABSA_Tweets.git

2. Download necessary dependencies

  - pip install -r requirements.txt

3. Access the configuration file in src/config.ini and write down your personal configuration parameters

### 2.2. The "load.py" file
Initially, there is nothing to modify as the configuration parameters comes from the config.ini file, where the user should put his/her own project configuration parameters

### 2.3. The "preprocess.py" file
- The configuration parameters are red from the config-ini file directly from the "environment variables" of the file. 
- There is two main dictionaries for the use case "target_aspects_dict" and "targets_entities_dict", **that should be modify directly** by the user, depending on his/her own use cases. The entities that appear in the dictionary, are those that are referred by the aspect dictionary, this is, any comment (tweet) about a given aspect, will be cross with their respective entities. Fo you use-case, you should **take care and modify both dictionaries, as they are related**.
