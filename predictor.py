import pandas as pd

# region model comparison imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
# endregion
from matplotlib import pyplot as plt
import seaborn as sns

# region classifiers with warnings
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
# endregion

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


class Predictor:
    def __init__(self, path, reduced=False):
        self.file_path = path
        self.significant_fields = [
            'MCP-3', 'OSM', 'IL8', 'CASP-8', 'CCL3', 'HGF', 'AXIN1', 'CCL4', 'CCL20', 'CXCL6',
            'IL18', 'VEGFA', 'SLAMF1', 'IL-10RA', 'response', 'sample_ID', 'condition'
        ]

        self.significant_fields_reduced = [
            'MCP-3', 'OSM', 'IL8', 'CASP-8', ' CCL3', 'sample_ID', 'condition', 'response',
            'sample_ID'
        ]

        self.partial_fields = [
            'MCP-3', 'OSM', 'IL8', 'CASP-8', 'CCL3', 'HGF', 'AXIN1', 'CCL4', 'CCL20', 'CXCL6',
            'IL18', 'VEGFA', 'SLAMF1', 'IL-10RA', 'response'
        ]
        self.proteins_values = [
            'MCP-3', 'OSM', 'IL8', 'CASP-8', 'CCL3', 'HGF', 'AXIN1', 'CCL4', 'CCL20', 'CXCL6',
            'IL18', 'VEGFA', 'SLAMF1', 'IL-10RA'
        ]

        self.proteins_values_reduced = [
            'MCP-3', 'OSM', 'IL8', 'CASP-8', 'CCL3',
        ]

        self.classifier_methods_names = [
            'KNeighborsClassifier', 'Linear_SVM', 'RBF SVM', 'DecisionTreeClassifier',
            'RandomForestClassifier', 'Neural Net', 'AdaBoostClassifier', 'Naive Bayes',
            'Gaussian Process', 'Logistic Regression'
        ]

        self.classifier_methods_list = [
            KNeighborsClassifier(4),
            SVC(kernel='linear', C=0.025, probability=True),
            SVC(gamma=2, C=1, probability=True),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(alpha=1, max_iter=10000),
            AdaBoostClassifier(),
            GaussianNB(),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            LogisticRegression()
        ]

        self.data_df, self.train_x, self.train_y, self.test_x, self.test_y = (
            self.get_data(responders_training=2, non_responders_training=2, reduced=reduced)
        )

    def do(self, show_roc=False, show_confusion_matrix=False):
        for i in range(len(self.classifier_methods_list)):
            model_name = self.classifier_methods_names[i]
            classifier_method = self.classifier_methods_list[i]
            print(f'=========={model_name}==========')
            y_pred = self.predict(classifier_method)
            print('Prediction', y_pred)
            print(f'Accuracy: {round(accuracy_score(self.test_y, y_pred) * 100, 2)}%')
            print()
            if show_roc:
                self.roc_predict(model_name, classifier_method)
            if show_confusion_matrix:
                self.get_confusion_matrix(model_name, y_pred)

    def get_data(self, responders_training=0, non_responders_training=0, reduced=False):
        fields = self.significant_fields_reduced if reduced else self.significant_fields
        proteins = self.proteins_values_reduced if reduced else self.proteins_values

        data = pd.read_csv(self.file_path, delimiter=';')
        data_df = data[data.columns.intersection(fields)]
        data_df = data_df[data_df['response'].isin(['R', 'NR'])]
        data_df.loc[data_df['response'] == 'R', 'response'] = 1
        data_df.loc[data_df['response'] == 'NR', 'response'] = 0
        data_df = data_df[data_df['condition'] == 'V1']
        data_df_resp = data_df[data_df['response'] == 1]

        # responders
        x_responders_df = data_df_resp[data_df_resp.columns.intersection(proteins)]
        x_responders_df = x_responders_df.values.tolist()
        y_responders_df = data_df_resp['response'].values.tolist()
        # non-responders
        data_df_non_resp = data_df[data_df['response'] == 0]
        x_non_responders_df = (
            data_df_non_resp[data_df_non_resp.columns.intersection(proteins)]
        )
        x_non_responders_df = x_non_responders_df.values.tolist()
        y_non_responders_df = data_df_non_resp['response'].values.tolist()

        half_up_x_responders_df = len(x_responders_df) // 2 + responders_training
        half_up_y_responders_df = len(y_responders_df) // 2 + responders_training
        half_up_x_non_responders_df = len(x_non_responders_df) // 2 + non_responders_training
        half_up_y_non_responders_df = len(y_non_responders_df) // 2 + non_responders_training

        print('========Data Distribution========')
        print('Training set responders: ', half_up_x_responders_df)
        print('Training set non responders: ', half_up_x_non_responders_df)
        print('Test set responders: ', len(x_responders_df) - half_up_x_responders_df)
        print('Test set non responders: ', len(x_non_responders_df) - half_up_x_non_responders_df)
        print()

        train_x = (x_responders_df[0:half_up_x_responders_df]
                   + x_non_responders_df[0:half_up_x_non_responders_df])
        train_y = (y_responders_df[0:half_up_y_responders_df]
                   + y_non_responders_df[0:half_up_y_non_responders_df])

        test_x = (x_responders_df[half_up_x_responders_df:]
                  + x_non_responders_df[half_up_x_non_responders_df:])
        test_y = (y_responders_df[half_up_y_responders_df:]
                  + y_non_responders_df[half_up_y_non_responders_df:])

        return data_df, train_x, train_y, test_x, test_y

    def predict(self, classifier_method):
        classifier_method.fit(self.train_x, self.train_y)
        prediction_y = classifier_method.predict(self.test_x)
        print('real value:', self.test_y)
        return prediction_y

    def get_confusion_matrix(self, model_name, y_pred):
        conmat = confusion_matrix(self.test_y, y_pred)
        val = np.mat(conmat)
        classnames = list(set(self.test_y))
        df_cm = pd.DataFrame(

            val, index=classnames, columns=classnames,

        )
        print(df_cm)
        df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'{model_name} Results')
        plt.show()

    def roc_predict(self, model_name, classifier_method):
        y_pred_prob = classifier_method.predict_proba(np.array(self.test_x))[:, 1]
        fpr, tpr, thresholds = roc_curve(self.test_y, y_pred_prob)
        sns.set()
        plt.plot(fpr, tpr)
        plt.plot(fpr, fpr, linestyle='--', color='k')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        auroc = np.round(roc_auc_score(self.test_y, y_pred_prob), 2)
        plt.title(f'{model_name} Model ROC curve; AUROC: {auroc}')

        plt.show()


if __name__ == '__main__':
    predictor = Predictor('data_predictive_model.csv')
    predictor.do()
