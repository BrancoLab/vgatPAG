from vgatPAG.database.db_tables import *
from vgatPAG.analysis.utils import get_mouse_session_data

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import norm as Normal
from tqdm import tqdm
from random import choices
from brainrender.colors import colorMap
from math import exp, sqrt, pi


"""
  Based on tutorial on naive bayes from machinelearningmastery.com
"""


"""
    To do the same in sklearn
    from sklearn.preprocessing import LabelEncoder

    categories = LabelEncoder().fit_transform(int_values)

    encodedX = pd.DataFrame({c:LabelEncoder().fit_transform(round(data[c])) for c in data.columns})
    features = [tuple(row.values) for i,row in encodedX.iterrows()]


    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model.fit(features, categories)

    predicted = model.predict(features)



"""

class NaiveBayes:
    def __init__(self, predictor, predicted, equalize_classes = True, min_samples_per_class=1000, nsamples=10000):
        """
            This class impleements a naive bayes classifier. 

            :param equalize_classes: bool. if True data are drop to make sure that all classes being predicted
                    appear with the same frequency in the dataset
            :param min_samples_per_class: int, if equalizing class frequency, the minimum number of samples
                    for the rarest class
            :param nsamples: if equalizing class frequency, take this number of samples (with replacement)
                    from each class

        """

        if not isinstance(predictor, pd.DataFrame):
            raise ValueError('Predictor should be dataframes')
        if not isinstance(predicted, dict):
            raise ValueError('Predicted should be a dictionary of one key')
        
        self.nsamples = nsamples

        # Split the data in training and test sets
        data = predictor.copy()
        self.first_col = data.columns[0]

        pred = list(predicted.keys())[0]
        data[pred] = predicted[pred]

        if equalize_classes:
            data = self.equalize_classes(data, min_samples_per_class)

        train, test = train_test_split(data, test_size=0.5)
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)

    def equalize_classes(self, data, min_samples_per_class):
        # Find which class has the lowest probability
        initial_len = len(data)
        model = self.summarize_data_by_class(data)

        # Make sure that every class has as many row as the rarest
        # ? By keeping the minimal number of samples:
        # nsamples = int(min([model[k][self.first_col].nrows for k in model.keys()]))
        # if nsamples < min_samples_per_class:
        #     raise ValueError('Not enough data left sorry, try reducing the number of bins')
        # with_replacement = False
        # ? By keeping the maximal number of samples
        
        # nsamples = int(max([model[k][self.first_col].nrows for k in model.keys()]))

        with_replacement = True

        dfs = []
        for label in model.keys():
            subdata = data.loc[data.category == label].reset_index(drop=True)
            # select N random rows
            dfs.append(subdata.sample(self.nsamples, replace=with_replacement))


        clean = pd.concat(dfs, ignore_index=True)
        # print(f'Equalizing classes frequency. Data length after cleaning: {initial_len} --> {len(clean)}')
        return clean

    def fit(self, plot_class_probas=False, **kwargs):
        """
            Fits the test data
        """
        model = self.summarize_data_by_class(self.train)

        if plot_class_probas:
            class_probas = {k: model[k][self.first_col].class_probability for k in model.keys()}

            if np.abs(sum(class_probas.values()) - 1.0)>.05: # some leeway beacuse of rounding errors
                raise ValueError("Something's fishy")

            # Plot the raw probability of each class in the dataset
            # It'd be better if these were roughly uniform
            f, ax = plt.subplots()
            ax.bar(class_probas.keys(), class_probas.values())
            ax.set(title='Probability of each class')
            plt.show()

        self.model = model
        return model

    def predict(self, maxrows = None, **kwargs):

        predictions, labels = [], []

        if maxrows is not None:
            if maxrows < len(self.test):
                idxs = choices(np.arange(len(self.test)), k=maxrows)
            else:
                idxs = np.arange(len(self.test))
        else:
            idxs = np.arange(len(self.test))

        for i in idxs:
            row = self.test.iloc[i]
            labels.append(row.category)
            row = row.drop('category')
            predictions.append(self.predict_row(row))

        self.labels, self.predicted =  labels, predictions

        return labels, predictions

    def fit_predict(self, **kwargs):
        self.fit(**kwargs)
        return self.predict(**kwargs)



    def evaluate(self, verbose=False):
        correct = len([predicted for actual, predicted in zip(self.labels, self.predicted)
                                if actual == predicted])
        perform =  correct / float(len(self.labels)) * 100

        class_probas = {k: round(self.model[k][self.first_col].class_probability, 2) for k in self.model.keys()}

        if verbose:
            print(f'Model predicted on test set with {round(perform, 3)}% accuracy')
            print(f'\nThese were the probability of each class in the train set: {class_probas}')
            print(f'The most frequent class had a probability of {max(list(class_probas.values()))}')
        return perform


    # ----------------------------------- Utils ---------------------------------- #    
    # Summarize data by class
    @staticmethod
    def summarize_data_by_class(data):
        """
            Gets the mean and std of each predicting variable (i.e. ROI activity)
            for each category of the predicted variable
        """
        summaries_by_class = {}
        for label in data['category'].unique():
            subdata = data.loc[data.category == label]

            summary = pd.DataFrame(dict(
                    mean = subdata.mean().values,
                    std = subdata.std().values, 
                    nrows = [int(len(subdata)) for i in np.arange(len(subdata.columns))],
                    class_probability = [len(subdata)/len(data) for i in np.arange(len(subdata.columns))]
            )).T
            summary.columns = subdata.columns
            summaries_by_class[label] = summary.drop('category', axis=1)
        return summaries_by_class

    def get_row_probabilities(self, row):
        """ 
            Compute the probability that each piece of data belongs to each class
                p(class|data) = p(X|class) * p(class)

                p(c == c0|data) = p(X0|c=c0)*p(X1|c=c0)*...*p(c=c0)

            Note, unlike in Bayes theorem we are not dividing by the denominator p(X), 
            so the result will not be a probability and the sum over probabilities
            will not be equal to one. Hoever for regression porpuses this is not
            necessary.
        """
        def compute_proba(x, mean, std):
            exponent = exp(-((x - mean)**2 / (2 * std**2)))
            return (1 / (sqrt(2 * pi) * std)) * exponent

        probabilities = dict()
        for class_val, summary in self.model.items():
            # For each class compute p(c=ci)
            # This is given by the number of rows with that class divided by the total number of rows
            
            probabilities[class_val] = self.model[class_val][self.first_col].class_probability
            
            # For each data variable, compute p(Xi|c=ci) and multiply that by the class probability
            for col in summary.columns:
                xgivenc = compute_proba(row[col], summary[col]['mean'], summary[col]['std'])
                probabilities[class_val] *= xgivenc

        return probabilities

    def predict_row(self, row):
        probabilities = self.get_row_probabilities(row)
        predicted_class = max(probabilities.items(), key=operator.itemgetter(1))[0]

        # Just to double check
        maxproba = np.max(list(probabilities.values()))
        if maxproba != probabilities[predicted_class]:
            raise ValueError('Something went wrong with get row prediction:\n'+
                    f'predicted: {predicted_class} - maxproba {maxproba} - probabilities {probabilities[predicted_class]}')

        return predicted_class


