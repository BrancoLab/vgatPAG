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
    def __init__(self, predictor, predicted, equalize_classes = True):
        if not isinstance(predictor, pd.DataFrame):
            raise ValueError('Predictor should be dataframes')
        if not isinstance(predicted, dict):
            raise ValueError('Predicted should be a dictionary of one key')
        
        # Split the data in training and test sets
        data = predictor.copy()

        pred = list(predicted.keys())[0]
        data[pred] = predicted[pred]

        if equalize_classes:
            data = self.equalize_classes(data)

        train, test = train_test_split(data, test_size=0.2)
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)

    def equalize_classes(self, data):
        # Find which class has the lowest probability
        initial_len = len(data)
        model = self.summarize_data_by_class(data)
        class_probas = {k: model[k][0].class_probability for k in model.keys()}
        rarest = min(class_probas.items(), key=operator.itemgetter(1))[0]

        # Make sure that every class has as many row as the rarest
        nsamples = len(data.loc[data.category == rarest])
        dfs = []
        for label in model.keys():
            subdata = data.loc[data.category == label].reset_index(drop=True)
            # select N random rows
            dfs.append(subdata.sample(nsamples))


        clean = pd.concat(dfs, ignore_index=True)
        # print(f'Equalizing classes frequency. Data length after cleaning: {initial_len} --> {len(clean)}')
        return clean

    def fit(self, plot_class_probas=False, **kwargs):
        """
            Fits the test data
        """
        model = self.summarize_data_by_class(self.train)

        if plot_class_probas:
            class_probas = {k: model[k][0].class_probability for k in model.keys()}

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
            idxs = choices(np.arange(len(self.test)), k=maxrows)
        else:
            idxs = np.arange(len(self.test))

        for i in idxs:
            row = self.test.iloc[i]
            predictions.append(self.predict_row(row))
            labels.append(row.category)

        self.labels, self.predicted =  labels, predictions

        return labels, predictions

    def fit_predict(self, **kwargs):
        self.fit(**kwargs)
        return self.predict(**kwargs)



    def evaluate(self, verbose=False):
        correct = len([predicted for actual, predicted in zip(self.labels, self.predicted)
                                if actual == predicted])
        perform =  correct / float(len(self.labels)) * 100

        class_probas = {k: round(self.model[k][0].class_probability, 2) for k in self.model.keys()}

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
                    nrows = [len(subdata) for i in np.arange(len(subdata.columns))],
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
            probabilities[class_val] = self.model[class_val][0].class_probability
            
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
            raise ValueError('Something went wrong with get row prediction')

        return predicted_class



if __name__ == '__main__':
    # ------------------------------- Get metadata ------------------------------- #
    # Get all mice
    mice = Mouse.fetch("mouse")

    # Get all sessions
    sessions = {m:(Session & f"mouse='{m}'").fetch("sess_name") for m in mice}

    # Get the recordings for each session
    recordings = {m:{s:(Recording & f"sess_name='{s}'" & f"mouse='{m}'").fetch(as_dict=True) for s in sessions[m]} for m in mice}
    
    
    # --------------------------------- Get data --------------------------------- #
    mouse = mice[0]
    sess = sessions[mouse][0]

    tracking, ang_vel, speed, shelter_distance, signals, _nrois, is_rec = get_mouse_session_data(mouse, sess, sessions)

    # Prep data
    data = pd.DataFrame({r:sig for r,sig in enumerate(signals)})
    data['xpos'] = tracking.x

    # Keep only timepoint during recording
    data = data[is_rec > 0].reset_index(drop=True)
    data = data[:-60] # drop the last few rows because sometimes there's some null values

    # ! testing stuff
    # data = data[data.xpos> 300]

    # Create categories based on the X position
    N_bins = 5
    categories = pd.cut(data['xpos'], bins= np.linspace(data.xpos.min()-1, data.xpos.max()+1, N_bins+1))
    int_values =  np.array([(i.left + i.right)/2 for i in categories.values])
    data = data.drop('xpos', axis=1)

    nb = NaiveBayes(data, dict(category=int_values))
    labels, predictions = nb.fit_predict(maxrows=None)
    nb.evaluate()



    # TODO think about unbiasing input dataset to make things easier to interpret
    # TODO or just use sklearn


