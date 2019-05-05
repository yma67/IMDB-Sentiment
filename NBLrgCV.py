import glob
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

xPositive = []
xNegative = []
xSeq = []
yContinuous = []
yDiscrete = []

for s in ['pos', 'neg']:
    files = glob.glob('./comp-551-imbd-sentiment-classification/train/'+ s +'/*.txt')
    files.sort(key=lambda x: int(x.split('/')[-1].split('_')[0]))
    for name in files:
        yContinuous.append(int(name.split('_')[1].replace('.txt', '')))
        xSeq.append(int(name.split('/')[-1].split('_')[0]))
        if s == 'pos':
            yDiscrete.append(1)
        else:
            yDiscrete.append(0)
        with open(name) as f:
            text = f.read()
            text.lower()
            text = re.sub(r"[^A-Za-z]", " ", text)
            if s == 'pos':
                xPositive.append(text)
            else:
                xNegative.append(text)


def bigramTokenizer(textr):
    textr = re.sub("[^a-zA-Z]", " ", textr.lower())
    return textr.split()


df = pd.DataFrame([[x, yd, yc] for x, yd, yc in zip(xPositive + xNegative, yDiscrete, yContinuous)], columns=['text', 'y discrete', 'y continuous'])


class NBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=66, dual=False, fit_intercept=True, penalty='l2', alpha=1):
        self.classifier = None
        self.X_ = None
        self.y_ = None
        self.r = None
        self.alpha = alpha
        self.C = C
        self.dual = dual
        self.fit_intercept = fit_intercept
        self.penalty = penalty

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        p = self.alpha + X[y == 1].sum(0)
        q = self.alpha + X[y == 0].sum(0)
        p_vn = p / (np.linalg.norm(p))
        q_vn = q / (np.linalg.norm(q))
        self.r = sparse.csr_matrix(np.log(p_vn / q_vn))
        X2 = X.multiply(self.r)
        self.classifier = LogisticRegression(C=self.C, dual=self.dual, fit_intercept=self.fit_intercept, penalty=self.penalty).fit(X2, y)
        self.X_ = X2
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        return self.classifier.predict(X.multiply(self.r))


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


unionPipeline = FeatureUnion([
    ('bigram', CountVectorizer(ngram_range=(2, 3), tokenizer=bigramTokenizer, max_df=0.65)),
    ('unigram', CountVectorizer(ngram_range=(1, 1), max_df=0.8))
])

commentAPipeline = Pipeline([
    ('union', unionPipeline),
    ('lrg', NBWrapper(C=66, alpha=1)) # 66 , loss='squared_hinge'
])

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['y discrete'], train_size=0.8, test_size=0.2, random_state=210)

gridSearch = GridSearchCV(commentAPipeline, {
    "lrg__C": [0.5, 66, 100, 100, 1000],
    'union__bigram__max_df': [0.5, 0.65, 0.8, 0.95],
    'union__unigram__max_df': [0.5, 0.65, 0.8, 0.95],
}, cv=2, verbose=10, n_jobs=-1)
gridSearch.fit(X_train, y_train)
report(gridSearch.cv_results_)
