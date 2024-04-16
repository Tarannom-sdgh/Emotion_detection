import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import string
import os
import pickle
import yaml
class EmotionPrediction:
    #Main class for emotion detection
    def __init__(self, config_file):
        ''' 
        input: config_file
        description: Initializes the class and loads or trains the model based on the configuration.
        '''
        self.train_data_path = None
        self.config_file = config_file
        self.label_cols = ['anger', 'disappointment', 'disgust', 'fear', 'joy', 'love','nervousness','surprise','neutral','sadness','amusement+excitement']
        self.vec= None
        self.trn_term_doc= None
        self.models = {}
        self.coefficients = {}
        self._load_train_data()
        
    def _load_train_data(self):
        config_data = self.read_config_file(self.config_file)
        self.train_data_path = config_data.get('train_data_path', None)
        if self.train_data_path is None:
            raise ValueError("Train data path not found in config file.")
        self.train_data = self._load_data_from_csv(self.train_data_path)
        self._preprocess()
   
    @staticmethod
    def tokenize(s):
        re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        return re_tok.sub(r' \1 ', s).split()
    
    def _preprocess(self):
        '''
        description: this function does preprocessing and vectorizing the text column in train and get it 
        ready for training the model.
        '''
        COMMENT = 'text'
        
        # Read configuration file
        config_data = self.read_config_file(self.config_file)
        train_flag = config_data.get('train', False)

        if train_flag:
            # Train new model
            self.vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=self.tokenize,
                                    min_df=3, max_df=0.9, strip_accents='unicode', use_idf=True,
                                    smooth_idf=True, sublinear_tf=True)
            self.trn_term_doc = self.vec.fit_transform(self.train_data[COMMENT])
            x = self.trn_term_doc
            # Save the vectorizer
            self.trn_term_doc_path = "models/vec_term_doc_filtereddata.pkl"
            with open(self.trn_term_doc_path, 'wb') as f:
                pickle.dump(self.vec, f)

            for label in self.label_cols:
                self.models[label], self.coefficients[label] = self._get_model(self.train_data[label], x)
            # Save the trained model
            self.save_model()
        else:
            with open("models/vec_term_doc_filtereddata.pkl", "rb") as f:
                self.vec=pickle.load(f)
            # Load existing model
            self.load_model()

    def _get_model(self, y, x):
        '''
        input: int label our model is learning and training on, vector of text from train dataframes
        output: LogisticRegression object
        description get back a logistic regression model trained on the transformed features and the transformation calculated based on the class distribution.
        '''
        x = self.trn_term_doc
        y = y.values
        r = np.log(self._pr(x, 1, y) / self._pr(x, 0, y))
        m = LogisticRegression(C=4, dual=False)
        x_nb = x.multiply(r)
        return m.fit(x_nb, y), r

    def _pr(self,x, y_i, y):
        '''
        input: int label
        output: float probability of each each label
        '''
        p = x[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def check_emotion(self, test_text):
        '''
        input: text string
        output: dictionary of probabilities for labels
        '''
        test_term_doc = self.vec.transform([test_text])
        preds = {}
        for label in self.label_cols:
            m = self.models[label]
            r = self.coefficients[label]
            prob = m.predict_proba(test_term_doc.multiply(r))[:, 1]
            preds[label] = prob[0]
        threshold=0.33
        keys_above_threshold = self.keys_above_threshold(preds, threshold)
        return keys_above_threshold
    
    @staticmethod
    def keys_above_threshold(preds, threshold):
        # Convert values of the dictionary to a numpy array
        all_values = np.array(list(preds.values()))
        
        # Calculate mean and variance
        mean = np.mean(all_values)
        variance = np.var(all_values)
        keys_above_threshold = []
        for key, value in preds.items():
            if np.any(value > threshold):
                keys_above_threshold.append(key)
            else:
                # Check if any number is higher than variance + mean
                above_threshold = value > (variance + mean)
                if np.any(above_threshold):
                    keys_above_threshold.append(key)
        return keys_above_threshold
    
    # Function to read the YAML file
    @staticmethod
    def read_config_file(filename):
        with open(filename, 'r') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return config_data
            except yaml.YAMLError as exc:
                print(exc)

    # Function to save the trained model
    def save_model(self):
        '''
        Saves the trained model.
        '''
        with open("models/emotion_predictor_model_filtereddata.pkl", "wb") as f:
            pickle.dump((self.models, self.coefficients), f)

    # Function to load the trained model
    def load_model(self):
        '''
        Loads the saved model.
        '''
        with open("models/emotion_predictor_model_filtereddata.pkl", "rb") as f:
            self.models, self.coefficients = pickle.load(f)
    
    def _load_data_from_csv(self, file_path):
        '''
        Loads data from a CSV file.
        '''
        return pd.read_csv(file_path) 
        
if __name__ == "__main__":
    import time
    
    # Get input text from the user
    text = input("Enter text: ")
    
    # Instantiate EmotionPrediction class with your config file
    emotion_predictor = EmotionPrediction("config.yml")

    # Predict emotions for the input text
    predicted_emotions = emotion_predictor.check_emotion(text)

    # Print the predicted emotions
    print("Predicted emotions for the text:", predicted_emotions)
    

# FOR USAGE:
# Assuming you have 'train' DataFrame with 'text' and labels for each emotion
# Set train and train_data_path config.yml
