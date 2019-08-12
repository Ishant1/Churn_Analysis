# Churn_Analysis
Project with complete process from Data loading to model creation and evaluation. Contains multiple helper classes and function to make pre-processing and model comparison easy and compact.

Below is an exaple of using those helper classes with the example dataset given in the repo

## Pre_Processing
The task in this project is done in a pipeline style and all helping classes for it are in *Processing.py*. Below is an example of the pipeline structure given in *classifier.py*

```
number_pipeline = Pipeline( steps = [ ( 'type_setter', type_setter() ),
                                      ( 'transformer', FeatureUnion(
                                                                    transformer_list = 
                                                                    [
                                                                        ( 'category',Pipeline([( 'select_type', select_type(type = 'category') ),
                                                                                              ( 'imputer', imputer(strategy = 'most_frequent')),
                                                                                              ( 'Encoder', Encoder(drop = 'first'))])
                                                                         ),
                                                                         ( 'numerical',Pipeline([( 'select_type', select_type(type = 'number') ),
                                                                                              ( 'imputer', imputer(strategy = 'mean')),
                                                                                              ( 'Standardizer', Standardizer())])
                                                                          )
                                                                    ]
                                                                    )
                                       )
                                     ] 
                           )
                           
```
All the classes here are custom classes allowing anyone to edit them easily and make them userspecific
Just like any other sklearn package this pipeline can also be fitted upon train set and be used upon testing data. 
Using column_extract the columns can be created for the pipeline numpy.array output

## Model creation & Evaluation
Class **evaluate_classifier** in *evaluate.py* file can be given a list of models and paramters to train over in the manner below:

```
models = [LinearSVC(),RandomForestClassifier(),AdaBoostClassifier()]

parameteres = {'LinearSVC':{'C':[0.5,1,2]},
              'RandomForestClassifier':{'n_estimators':[10,20,25], 'min_samples_split':[2,5,10]},
              'AdaBoostClassifier':{'n_estimators' : [50,75,100],'learning_rate' : [0.5,1,2]}
               }
 ```

The class can be created with above param and be fitted with training data. These models can be further evaluated with testing data giving out metrics like auc, f1_score, precison and recall
```
evaluator = evaluate_classifier(models, parameters = parameteres)
evaluator.fit(X_num_Train_c, y_train)
evaluator.evaluate_and_score(X_num_Test_c,y_test)
```
Then these metrics can be compared with a precision-recall curve which can be obtained through **evaluator.plot_results()** looks like below

![Precision-Recall Curve!](/churn/P_R_curve.png "Precision-Recall Curve")

---

Thus in this manner using pipeline based pre-processing can bring structure to data processing and compact model creation and evaulation can help reduce time to predict!!





