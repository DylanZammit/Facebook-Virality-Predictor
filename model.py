import re
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from nltk.corpus import stopwords

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

from imblearn.over_sampling import RandomOverSampler

np.random.seed(42)

DummyClassifier = partial(DummyClassifier, strategy='stratified')

pd.set_option('display.max_rows', None)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

BASE_DIR = os.path.dirname(__file__)

url_regex = r"(http:\/\/www.|https:\/\/www.|http:\/\/|https:\/\/)?[a-zA-Z0-9]+([-.]{1}[a-zA-Z0-9]+)*\.[a-z][a-zA-Z]{1,7}(:[0-9]{1,5})?(\/[^ \n\r\)]+)?"

days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

def clean(orig):
    if not orig: return ''
    x = orig
    x = x.lower()

    punctuation = [r',', r'.']

    for p in punctuation:
        x = x.replace(p, '')
    breakers = [r'\n', r'\t']
    for p in breakers:
        x = x.replace(p, ' ')

    for dow in days_of_week:
        x = x.replace(dow, 'DAY_OF_WEEK')

    x = re.sub(r'\d+%', ' PCT_HERE ', x)
    x = re.sub(r'\d+(?:km|m)', ' DIST_HERE ', x)
    x = re.sub(r'(?:19|20)\d\d\D', ' YEAR_HERE ', x)
    x = re.sub(r'(?:€|\$)\d+', ' MONEY_HERE ', x)
    x = re.sub(r'\d+', ' NUMBER_HERE ', x)
    x = re.sub(url_regex, ' LINK_HERE ', x)
    x = re.sub(r'"', r' QUOTATION_HERE ', x)
    # not needed if stemming is used
    x = re.sub(r"'s\b", r'', x)  # dog, dog's  -> dog
    # x = re.sub(r"s\b", r'', x)   # dog, dogs   -> dog
    x = re.sub(r'[^a-z\s]', r'', x)
    x = re.sub(r'\b\w\b', r' ', x)
    x = re.sub(r'\s+', r' ', x)
    x = x.strip()
    return x


def save_ml(obj, fn='ml_model.pkl'):
    base_dir = os.path.dirname(__file__)
    fn = os.path.join(base_dir, fn)
    try:
        with open(fn, 'wb+') as f:
            print(f'saving to {fn}...', end='', flush=True)
            pickle.dump(obj, f)
    except Exception as e:
        print(f'FAILED TO SAVE: {e}')
    else:
        print(f'saved!')


def load_ml(fn='ml_model.pkl'):
    base_dir = os.path.dirname(__file__)
    fn = os.path.join(base_dir, fn)
    with open(fn, 'rb') as f:
        print(f'loading {fn}...', end='', flush=True)
        model = pickle.load(f)
    print(f'loaded {fn}')
    return model


class LemmaTokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


stemmer = LancasterStemmer()


class StemmedCountVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def get_data(viral_threshold=20):
    sheet_id = r'1mGNZX6qb7hMnKa9va_cyTjJm-B9RGV225U_JzEZryHw'
    sheet_name = 'Sheet1'
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    data = pd.read_csv(url)
    data['engagement'] = np.where(data.tot_engagement < viral_threshold, 0, 1)
    return data


class MLOTP:
    
    def __init__(
            self,
            classifier='SGD', 
            use_stop_words=True,
            oversample=True,
            stem=True,
            **kwargs
            ):
        """
        kwargs: go to classifier
        """

        stop_words = stopwords.words('english') if use_stop_words else []

        vectorizer = StemmedCountVectorizer if stem else TfidfVectorizer

        self.count_vect = vectorizer(
            min_df=2,
            max_df=0.8,
            stop_words=stop_words,
            analyzer='word',
            ngram_range=(1, 2),
            # tokenizer=LemmaTokenizer(),
            # preprocessor=clean,
        )
        self.stop_words = stop_words
        self.oversample = oversample

        self.clf_name = classifier

        classifier_map = {
            'MNB': MultinomialNB,
            'BNB': BernoulliNB,
            'SGD': SGDClassifier,
            'PPT': Perceptron,
            'PA':  PassiveAggressiveClassifier,
            'DU': DummyClassifier,
        }

        try:
            self.clf = classifier_map[classifier](**kwargs)
        except KeyError:
            raise ValueError(f'Classifier {classifier} not supported')

        self.stats = {
            'n_train': 0,
            'n_train_hist': [],
            'accuracy': 0,
            'accuracy_hist': [],
            'total_fit_time': 0,
        }

        data = get_data()
        # data.tot_engagement.quantile(0.8) ~35 engagement

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.drop(['engagement'], axis=1),
            data.engagement,
            test_size=0.3,
            random_state=42
        )
        self.X_test_orig = self.X_test.copy()

        self.test_setup()

    def test_setup(self):
        self.X_test['message_orig'] = self.X_test.Message
        self.X_test.Message = self.X_test.Message.apply(clean)
        self.X_test = self.parse(self.X_test)

    def parse(self, data):
        data['day_of_week'] = data['Created time'].apply(lambda x: pd.Timestamp(x)).dt.dayofweek
        data['hour_of_day'] = data['Created time'].apply(lambda x: pd.Timestamp(x)).dt.hour
        data['post_length'] = data.Message.apply(lambda x: len(x.split()))

        day_of_week = pd.get_dummies(data.day_of_week)
        day_of_week = day_of_week[sorted(day_of_week.columns)]
        day_of_week = np.array(day_of_week)

        # hour_of_day = pd.get_dummies(data.hour_of_day)
        # hour_of_day = hour_of_day[sorted(hour_of_day.columns)]
        # hour_of_day = np.array(hour_of_day)

        x = data.Message
        post_length = data.post_length.to_numpy().reshape(-1, 1)

        try:
            # for testing
            x = self.count_vect.transform(x).toarray()
        except: # escape specific error...
            # for training
            x = self.count_vect.fit_transform(x).toarray()

        if 0:
            for w in self.count_vect.get_feature_names_out():
                print(w)
            breakpoint()
        # also include extra features such as post length and create hour of post
        x = np.concatenate((
                x,
                # day_of_week,
                # hour_of_day,
                # post_length,
            ), axis=1)
        return x

    def train(self):
        tick = time.time()
        self.X_train.Message = self.X_train.Message.apply(clean)

        self.X_train = self.parse(self.X_train)

        # over/under sample
        if self.oversample:
            # see also RandomUnderSampler, ADASYN, InstanceHardnessThreshold, SMOTEENN
            print('oversampling...')
            self.X_train, self.y_train = RandomOverSampler(random_state=0).fit_resample(self.X_train, self.y_train)

        print('fit model...')
        self.clf.fit(self.X_train, self.y_train)
        # self.clf.partial_fit(x, y, classes=[0, 1]) # if run in batches use this
        tock = time.time()

        n_train = len(self.X_train)
        y_pred = self.clf.predict(self.X_test)

        accuracy = round(accuracy_score(self.y_test, y_pred), 4)
        f1 = round(f1_score(self.y_test, y_pred), 4)
        recall = round(recall_score(self.y_test, y_pred), 4)
        precision = round(precision_score(self.y_test, y_pred), 4)

        self.stats['n_train'] += n_train
        self.stats['n_train_hist'].append(n_train)
        self.stats['accuracy'] = accuracy
        self.stats['f1_score'] = f1
        self.stats['precision'] = precision
        self.stats['recall'] = recall
        self.stats['total_fit_time'] += tock-tick
        print(self)
        return self

    def predict(self, caption, created_time, df=None, threshold=None):
        raise NotImplementedError()
        if df is None:
            if not isinstance(caption, list):
                caption = [caption]
                created_time = [created_time]
            df = pd.DataFrame({
                'Message': caption,
                'Created time': created_time,
            }, index=range(len(caption)))

        df_parsed = df.copy()
        df_parsed.Message = df_parsed.Message.apply(clean)
        x = self.parse(df_parsed)
        df['prob'] = self.clf.predict_proba(x)[:, 1]
        df['pred'] = (df.prob >= threshold) * 1 if threshold else self.clf.predict(x)
        return df

    def predict_test(self, threshold=0.5):
        self.X_test_orig['prob'] = self.clf.predict_proba(self.X_test)[:, 1]
        self.X_test_orig.prob = self.X_test_orig.prob.apply(lambda x: round(x, 2))
        self.X_test_orig['pred'] = (self.X_test_orig.prob >= threshold) * 1
        self.X_test_orig['engagement'] = self.y_test
        return self.X_test_orig

    def top_words(self, n=50):
        '''
        Only works for certain classifiers
        :param n: top n words
        '''

        # handle case if extra features are included!!
        # They should always be the last features, but double check
        words_idx = self.clf.coef_[0].argsort()

        features = self.count_vect.get_feature_names_out()
        # n_features = len(features)

        influential_words = np.take(features, words_idx[-n:])
        boring_words = np.take(features, words_idx[:n])

        return boring_words, influential_words

    def plot(self):
        y_score = self.clf.decision_function(self.X_test)

        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        precision, recall, _ = precision_recall_curve(self.y_test, y_score, pos_label=self.clf.classes_[1])
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        ax1.plot([(0, 0), (1, 1)])
        plt.show()

    @property
    def accuracy(self):
        return self.stats['accuracy']

    @property
    def n_train(self):
        return self.stats['n_train']

    @property
    def fit_time(self):
        return self.stats['total_fit_time']

    def __len__(self):
        return self.n_train

    def __repr__(self):
        out = '*'*10+' SUMMARY ' + '*'*11 + '\n'
        for k, v in self.stats.items():
            out += f'{k}: {v}\n'
        out += '*'*30+'\n'
        return out


if __name__ == '__main__':

    print('Welcome to Facebook Post react predictor!')

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load model", action='store_true')
    parser.add_argument("--save", help="save model", action='store_true')
    parser.add_argument("--oversample", help="oversample", action='store_true')
    parser.add_argument("--stem", help="use stemming", action='store_true')
    parser.add_argument("--classifier", help="classified [def=SGD]", default='SGD')
    parser.add_argument("--threshold", help="threshold prob must exceed for label to be 1", default=0.5, type=float)
    parser.add_argument("--viral_threshold", help="engagement > viral_threshold => 1, else 0", default=20, type=float)
    args = parser.parse_args()
    oversample = args.oversample
    load = args.load
    save = args.save
    stem = args.stem
    T = args.threshold

    classifier = args.classifier
    kwargs = {
        'loss': 'log_loss',
        'penalty': 'l1',
    }

    if load:
        ml = load_ml()
        print('setting up test data...', flush=True)
        ml.test_setup()
    else:
        print('Creating new model...')
        ml = MLOTP(
            classifier=classifier,
            oversample=oversample,
            stem=stem,
            **kwargs
        )
        ml.train()
        if save:
            fn = 'FB_model.pkl'
            save_ml(ml, fn)
    print('predicting test data...', flush=True)
    df_test = ml.predict_test(T)
    try:
        neg, pos = ml.top_words()
        print('Non-Influential features')
        print(neg)
        print('Influential features')
        print(pos)
    except AttributeError:
        pass

    try:
        ml.plot()
    except AttributeError:
        pass

    A = df_test[df_test.engagement == 1].pred.mean()
    B = df_test[df_test.engagement == 0].pred.mean()

    # print(ml)
    print(f'Sample size = {ml.n_train}')
    print('Of Viral posts, ML agreed with {:.1f}%'.format(A*100))
    print('Of NON-Viral posts, ML agreed with {:.1f}%'.format(100-int(B*100)))
    print('Out-of-sample accuracy = {:.1f}%'.format(ml.accuracy*100))

    df_test.to_csv('predictions.csv')

    exit()
    from datetime import datetime
    captions = ['This is some sample caption. A prediction of virality for this caption example will be given.']
    df_single = ml.predict(captions, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(df_single)
