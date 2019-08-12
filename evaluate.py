import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import GridSearchCV



# Helper class to evaluate multiple classes togather and function to compare them and also choose best prarameters for each

class evaluate_classifier(BaseEstimator,TransformerMixin):


    def __init__(self,models,parameters={}):

        #lis of models to be used to evaluate
        self.models = models
        self.model_names = [str(x).split('(')[0] for x in self.models]

        self.parameteres = parameters

        #All the results and models will be stored in variable below
        self.result = {}

    
    
    
    
    #Fits the models given as a list and also searches best parameters for each through GridSeach
    def fit(self,X_train,y_train,cv_=5):
        #looping through all the models in the list
        for model in self.models:

            model_name = str(model).split('(')[0]

            print('Fitting '+model_name)
            if model_name in self.parameteres:
                grid_model = GridSearchCV(model,param_grid=self.parameteres[model_name], cv=cv_)
                grid_model.fit(X_train,y_train)

                self.result.update({model_name : {'model':grid_model,
                                                  'best_fit_param':grid_model.best_params_,
                                                  'cv_best_score':grid_model.best_score_}})
                print('Fitted {} with best score:{:.3f}'.format(model_name,grid_model.best_score_))
            else:
                model_ = model
                model_.fit(X_train,y_train)

                self.result.update({model_name : {'model':model_}})

                print('Fitted {}'.format(model_name))

    #Uses all the fitted models from before and create different metrics with test data for cmoparision 
    def evaluate_and_score(self, X_test, y_test):
        for model_name in self.model_names:
            model = self.result[model_name]['model']
            f1_score_ = f1_score(y_test, model.predict(X_test))
            # Generate the P-R curve
            try:
                y_prob = model.decision_function(X_test)
            except:
                y_prob = model.predict_proba(X_test)[:,1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)

            auc_ = auc(recall,precision)

            self.result[model_name].update({'f1_score':f1_score_,
                                            'auc': auc_,
                                            'precision':precision,
                                            'recall':recall})

    #Helper function to plot the results from all models on a single graph
    def plot_results(self):
        fig = plt.figure(figsize=(6, 6))

        for model_name in self.model_names:
            recall = self.result[model_name]['recall']
            precision = self.result[model_name]['precision']
            plt.plot(recall, precision, label=model_name)

        plt.title('Precision-Recall Curves')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.legend(loc='lower left')

        # Let matplotlib improve the layout
        plt.tight_layout()

        plt.show()

        plt.close()

    
    #Outputs the respective metrics of each model in a table for easy comparision
    def results(self):

        ouput = []
        for x in self.model_names:
            if 'f1_score' in self.result[x]:
                ouput.append([x,self.result[x]['f1_score'],self.result[x]['auc']])
            else:
                return ValueError('Evaluate the models first with evaluate_and_score')

        return pd.DataFrame(ouput,columns = ['models','f1_score','auc'])


