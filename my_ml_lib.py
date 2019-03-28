import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.plotting import table
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class DimensionalityReductionTools:
    @staticmethod
    def pca(data, conserve, top_eig_vecs=None):
        """
        Performs Principal Component Analysis(PCA) dimensionality reduction over data.

        data: data to project in reduced dimensions.
        conserve: amount of eigen energy to be converved (value is between 0 to 1).
        top_eig_vecs: if given a numpy array of eigen vectors stacked horizontally, 
                      the data will be projected using these eigen vectors.
                      
        return: projection of data in reduced dimesion and the eigen vector's stack used to project data.
        """
        
        if top_eig_vecs is None:
            cov_mat = np.cov(data.T)
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            eig_vals = np.real(eig_vals)
            
            # Find indexes of largest eigen values
            top_eig_indxes = np.argsort(-eig_vals)

            # Get the top-k eigen vectors corresponding to top-k eigen values 
            # to conserve required eigen energy
            energy = 0
            total_energy = np.sum(eig_vals)
            top_eig_vecs = []
            k = 0
            while energy/total_energy < conserve:
                energy += eig_vals[top_eig_indxes[k]]
                top_eig_vecs.append(eig_vecs[:, top_eig_indxes[k]])
                k += 1

            top_eig_vecs = np.array(top_eig_vecs)
            # Transpose to get horizontally stacked eigen vectors
            top_eig_vecs = np.real(top_eig_vecs.T)
            
        mean = np.average(data, axis=0)
        data = data - mean
        return np.dot(data, top_eig_vecs), top_eig_vecs
    
    @staticmethod
    def lda(data_x, data_y, n_components=None, top_eig_vecs=None):
        """
        Performs Linear Discriminant Analysis(LDA) dimensionality reduction over data.
        
        data_x: Features numpy array.
        data_y: labels corresponding feature vectors.
        n_components: Number of components in which the data will be reduced.
                      if None, n_components = n_classes - 1.
        top_eig_vecs: if given a numpy array of eigen vectors stacked horizontally, 
                      the data will be projected using these eigen vectors.
                      
        return: projection of data in reduced dimesion and the eigen vector's stack used to project data.
        """
        if top_eig_vecs is None:
            classes = np.sort(np.unique(data_y))

            if n_components is None:
                n_components = len(classes) - 1

            data_mean = np.mean(data_x, axis=0).reshape((data_x.shape[1], 1))

            Sw = np.zeros((data_x.shape[1], data_x.shape[1]))
            Sb = np.zeros((data_x.shape[1], data_x.shape[1]))

            for c in classes:
                data_xc = data_x[data_y == c]
                mean_c = np.mean(data_xc, axis=0)
                sw_c = np.zeros((data_x.shape[1], data_x.shape[1]))

                mean_c = mean_c.reshape((data_x.shape[1], 1))

                for row in data_xc:
                    row = row.reshape((data_x.shape[1], 1))
                    sw_c += np.dot((row - mean_c),(row - mean_c).T)
                Sw += sw_c

                Sb += len(data_xc) * np.dot((mean_c - data_mean), (mean_c - data_mean).T)

            Sw_inv = np.linalg.pinv(Sw)
            eig_vals, eig_vecs = np.linalg.eig(np.dot(Sw_inv, Sb))
            eig_vals = np.real(eig_vals)

            # Find indexes of largest eigen values
            top_eig_indxes = np.argsort(-eig_vals)

            top_eig_vecs = []
            for k in range(0, n_components):
                top_eig_vecs.append(eig_vecs[:, top_eig_indxes[k]])

            top_eig_vecs = np.array(top_eig_vecs)
            # Transpose to get horizontally stacked eigen vectors
            top_eig_vecs = np.real(top_eig_vecs.T)

        return np.dot(data_x, top_eig_vecs), top_eig_vecs

    
class DataManipulationTools:
    @staticmethod
    def split_data(data, split, shuffle=True, random_state=None):
        """
        data: numpy array to be split in the given ratios.
        split: ratio between 0 to 1 to split the data. Eg. 0.7 means 70% - 30% data split.
        shuffle: if True, shuffles the data randomly. (default: True)
        random_state: seed for the random number generator. (default: None)
        
        return: data split in 2 parts of sizes (split * len(data), (1-split) * len(data))
        """

        if random_state:
            np.random.seed(random_state)
        if shuffle:
            data = np.random.permutation(data)

        idx = int(np.ceil(data.shape[0] * split))
        return (data[:idx], data[idx:])
    
    def k_folds(data, k=5, shuffle=False, random_state=None):
        """
        Perform K-Folds on given data

        data: data to be split into k folds.
        k: number of folds to be generated. (default: 5)
        shuffle: True if you want to shuffle the data before forming the folds. (default: False)
        random_state: integer seed for random number generator.

        return: a generator to give training and validation folds for k iterations.
        """
        fold_size = int(np.ceil(len(data) / k))

        if random_state:
            np.random.seed(random_state)
        if shuffle:
            data = np.random.permutation(data)

        folds = []
        start = 0
        end = fold_size
        for i in range(k):
            folds.append(data[start:end])
            start = end
            end += fold_size

        folds = np.array(folds)
        for i in range(k):
            if i == 0:
                yield np.concatenate(folds[1:]), folds[0]
            elif i == k-1:
                yield np.concatenate(folds[:k-1]), folds[k-1]
            else:
                yield np.concatenate(np.concatenate((folds[:i], folds[i+1:]))), folds[i]


class MetricTools:
    @staticmethod
    def accuracy(y, y_hat):
        """
        y [np array]: actual labels
        y_hat [np array]: predicted labels
        
        return: accuracy between 0 and 1
        """
        return np.sum(y == y_hat) / len(y)
    
    @staticmethod
    def prec_recall(y, y_hat, nclasses):
        """
        y [np array]: actual labels
        y_hat [np array]: predicted labels
        nclasses [integer]: number of classes in the dataset.
        
        return: precision, recall
        """
        cm = MetricTools.confusion_matrix(y, y_hat, nclasses)
        
        rec = cm[0,0] / np.sum(cm[0,:])
        prec = cm[0,0] / np.sum(cm[:,0])
        
        return prec, rec
    
    @staticmethod
    def confusion_matrix(y, y_hat, nclasses):
        """
        y [np array]: actual labels [values between 0 to nclasses-1]
        y_hat [np array]: predicted labels [values between 0 to nclasses-1]
        nclasses [integer]: number of classes in the dataset.
        
        return: confusion matrix of shape [nclasses, nclasses]
        """
        y = y.astype(np.int64)
        y_hat = y_hat.astype(np.int64)

        conf_mat = np.zeros((nclasses, nclasses))

        for i in range(y_hat.shape[0]):
            true, pred = y[i], y_hat[i]
            conf_mat[true, pred] += 1

        return conf_mat
    
    @staticmethod
    def roc_curve(probs, test_y, label, is_log_prob=True):
        """
        probs: Default Log-Posteriors log(P(C_label/x)) for class label. 
               Could be normal probs if is_log_prob = False.
        test_y: actual labels of test data.
        label: Class for which ROC is to be formed.
        is_log_prob: Tells if probs is log-prob or not
        
        return: False Positive Rate (FPR), True Positive Rate (TPR) for the given values
        """
        thresholds = np.linspace(0, 1, num=100)

        x = []
        y = []

        if is_log_prob:
            probs = np.exp(probs)
        
        min_val = np.min(probs)
        max_val = np.max(probs)

        for thresh in thresholds:
            conf_mat = np.zeros((2,2))
            pred = None
            actual = None
            for i in range(test_y.shape[0]):
                if (probs[i] - min_val) / (max_val - min_val) > thresh:
                    pred = 0
                else:
                    pred = 1

                if test_y[i] == label:
                    actual = 0
                else:
                    actual = 1
                conf_mat[actual, pred] += 1

            fpr = conf_mat[1, 0] / np.sum(conf_mat[1, :])
            tpr = conf_mat[0, 0] / np.sum(conf_mat[0, :])

            x.append(fpr)
            y.append(tpr)

        return x, y
       

class PlotTools:
    @staticmethod
    def confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(7,7), path=None, filename=None):
        """
        cm: confusion matrix to be plotted.
        classes: array of labels or class names.
        title: title of the confusion matrix.
        cmap: color of the plot matrix.
        figsize: tupple (width, height) representiong size of the plot.
        path: destination where the plot image will be saved.
        filename: name to save the file with on the specified path. (if None, title is used)
        
        # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        cm = cm.astype(np.int64)
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if path:
            if filename is None:
                plt.savefig(path + title + '.png')
            else:
                plt.savefig(path + filename + '.png')
        plt.show()
                
    @staticmethod
    def roc_curves(rocs, title, figsize=(8,5), path=None, filename=None):
        """
        rocs: Dictionary of the form {'label': [FPR, TPR]} containing multiple ROC values.
              where FPR = False positive rate; TPR = True positive rate
              label = Title for which the ROC values are given
        title: Title of the plot
        figsize: tupple (width, height) representiong size of the plot.
        path: destination where the plot image will be saved.
        filename: name to save the file with on the specified path. (if None, title is used)
        """
        plt.figure(figsize=figsize)
        for l, roc in rocs.items():
            plt.plot(roc[0], roc[1], label=l)

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim((-0.05, 1.08))
        plt.title(title)
        plt.legend()
        
        if path:
            if filename is None:
                plt.savefig(path + title + '.png')
            else:
                plt.savefig(path + filename + '.png')
        plt.show()

    @staticmethod
    def table(data, row_index, col_index, title, figsize=(8,3), col_widths=[0.5], path=None, filename=None):
        """
        Plots the data in tabular format.
        
        data: 2d array data for the table to be plotted.
        row_index: Headers for Rows of the table.
        col_index: Headers for Columns of the table.
        title: Title of the table.
        figsize: tupple (width, height) representiong size of the plot.
        col_widths: width of each column in the table.
        path: destination where the plot image will be saved.
        filename: name to save the file with on the specified path. (if None, title is used)
        """
        df = pd.DataFrame(data)
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, frame_on=False) 
        ax.xaxis.set_visible(False)  
        ax.yaxis.set_visible(False)
        plt.title(title)
        table(ax, df, loc='upper right', rowLabels=row_index, colLabels=col_index, colWidths=col_widths)
        if path:
            if filename is None:
                plt.savefig(path + title + '.png')
            else:
                plt.savefig(path + filename + '.png')
        plt.show()

        
class Ada_Boost:
    def __init__(self):
        self.classifiers = []
        self.alphas = []
        self.le = None
        
    def fit(self, train, n, tree_max_depth=2, tree_max_nodes=5, random_state=None):
        """
        n: Number of boosting iterations i.e. number of classifiers.
        train: Training data set, where last column are the labels 
               corresponding to each data point.
        tree_max_depth: maximum tree depth of a weak learner.
        tree_max_nodes: maximum number of leaf nodes possible in a weak learner.
        random_state: seed for random generator.

        return: (predictions, accuracy, error rate) over training set
        """
        
        self.le = LabelEncoder()
        self.le.fit(train[:, -1])
        
        X = train[:, :-1]
        y = self.le.transform(train[:, -1])
        
        # number of data points
        n_data = len(X)
        n_classes = len(self.le.classes_)
        
        weights = np.ones((n_data)) / n_data
        ensemble_train_error = 1
        
        for i in range(n):
            self.classifiers.append(DecisionTreeClassifier(max_depth=tree_max_depth, 
                                                           max_leaf_nodes=tree_max_nodes,
                                                           random_state=random_state))
            
            self.classifiers[i].fit(X, y, sample_weight=weights)
            
            preds = self.classifiers[i].predict(X)
            
            I = np.ones((n_data))
            I[preds == y] = 0
            
            error = np.sum(np.multiply(weights, I)) / np.sum(weights)
            
            ensemble_train_error *= 2 * np.sqrt(error * (1 - error))
            
            alphai = np.log((1 - error) / error) + np.log(n_classes - 1)
            self.alphas.append(alphai)
            
            I[preds == y] = -1
            
            # Update weights
            weights = np.multiply(weights, np.exp(alphai * I))
            # normalize weights
            weights = weights / np.sum(weights)
        
        y_hat = self.predict(X)
        train_acc = MetricTools.accuracy(train[:, -1], y_hat)
        
        return y_hat, train_acc, ensemble_train_error
    
    def predict(self, test):
        """
        test: data for which labels are to be predicted.
        
        return: numpy array of label corresponding to each test point.
        """
        # get a matrix of shape (test_points, n_classes)
        mat = np.zeros((len(test), len(self.le.classes_)))
        
        for i, clf in enumerate(self.classifiers):
            preds = clf.predict(test)
            
            for j, label in enumerate(preds):
                mat[j, label] += self.alphas[i]
        
        return self.le.inverse_transform(np.argmax(mat, axis=1))

    
class Bagging:
    def __init__(self):
        self.classifiers = []
        self.le = None
        
    def fit(self, train, n, tree_max_depth=2, tree_max_nodes=5, random_state=None):
        """
        n: Number of bagging iterations i.e. number of classifiers.
        train: Training data set, where last column are the labels 
               corresponding to each data point.
        tree_max_depth: maximum tree depth of a weak learner.
        tree_max_nodes: maximum number of leaf nodes possible in a weak learner.
        random_state: seed for random generator.

        return: (predictions, accuracy, error rate) over training set
        """
        
        self.le = LabelEncoder()
        self.le.fit(train[:, -1])
        
        X = train[:, :-1]
        y = self.le.transform(train[:, -1])
        
        # number of data points
        n_data = len(X)
        n_classes = len(self.le.classes_)
        
        for i in range(n):
            self.classifiers.append(DecisionTreeClassifier(max_depth=tree_max_depth, 
                                                           max_leaf_nodes=tree_max_nodes,
                                                           random_state=random_state))
            
            data_index = np.random.randint(0, n_data, n_data)
            train_X_i = X[data_index]
            train_y_i = y[data_index]
            
            self.classifiers[i].fit(train_X_i, train_y_i)
            
        y_hat = self.predict(X)
        train_acc = MetricTools.accuracy(train[:, -1], y_hat)
        
        return y_hat, train_acc, 1 - train_acc
    
    def predict(self, test):
        """
        test: data for which labels are to be predicted.
        
        return: numpy array of label corresponding to each test point.
        """
        # get a matrix of shape (test_points, n_classes)
        mat = np.zeros((len(test), len(self.le.classes_)))
        
        for i, clf in enumerate(self.classifiers):
            preds = clf.predict(test)
            
            for j, label in enumerate(preds):
                mat[j, label] += 1
        
        return self.le.inverse_transform(np.argmax(mat, axis=1))
