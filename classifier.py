#following are imported from the evaluate class given below
##import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#Different helper classes to help in bulding preprocessing pipeline
from Preprocessing import *

#imports classe which can be used to evaluate multiple models with parameter selection option
from evaluate import *

#function to split data into train/test sets
from sklearn.model_selection import train_test_split

#All the model packages form the sklearn, different packages can be used togather 
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Correlation function for finding dataset correlations
from scipy.stats import pearsonr




# [OPTIONAL] Seaborn makes plots nicer
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 175)
'''
try:
    import seaborn
except ImportError:
    pass
'''
# =====================================================================

file_path = "Data/telco-customer.csv"

def download_data():

    frame = pd.read_csv(
        file_path,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',

        # Specify the file encoding
        encoding='utf-8',

        # Specify the separator in the data
        sep=',',            

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,

        #generate first line as header
        header=0   

        )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame

Data = download_data()

# =====================================================================



#Splitting the Data into X and Y and Trian and test. Also making Y numeric
def prepare_data(Data,test_size=0.7):
    X = Data.iloc[:,:-1]
    y = Data.iloc[:,-1]
    #Checking if target is numerical or not
    if y.dtype =='O':
        y = y.astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(Data,test_size=0.3)




#Pipeline for basic null value handeling and standardization
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

#As sklearn  pipeline ouputs numpy array following function extract column names from the steps of pipeline
def column_extract(pipeline,Dataset):
    category_labels = []

    column_records = number_pipeline.named_steps['type_setter'].record

    for i,x in column_records.items():
        if x == 'category':
            category_labels+=[i]

    cat_col = list(pipeline.named_steps['transformer'].transformer_list[0][1][2].get_feature_names())
    num_col = list(number_pipeline.named_steps['transformer'].transformer_list[1][1][2].stats_.index.values)

    columns_ = cat_col + num_col

    for i,x in enumerate(columns_):
            numbers = "".join(list(filter(str.isdigit, x)))
            if len(numbers)>0:
                key = int(numbers)
                val = x.split('_')[1]
                columns_[i] = category_labels[key]+'_'+val

    return pd.DataFrame(Dataset,columns = columns_)


#Train and test set after the pre-processing 
X_num_Train = number_pipeline.fit_transform(X_train)
X_num_Test = number_pipeline.transform(X_test)

#Pandas Dataframes with column names attached
X_num_Train_c = column_extract(number_pipeline,X_num_Train)
X_num_Test_c = column_extract(number_pipeline,X_num_Test)


#list of models to be evaluated for comparision
models = [LinearSVC(),RandomForestClassifier(),AdaBoostClassifier()]

#list of parameters for the above models to itterated with GVSearch to find best combo
parameteres = {'LinearSVC':{'C':[0.5,1,2]},
              'RandomForestClassifier':{'n_estimators':[10,20,25], 'min_samples_split':[2,5,10]},
              'AdaBoostClassifier':{'n_estimators' : [50,75,100],'learning_rate' : [0.5,1,2]}
               }



#make the classification class object
evaluator = evaluate_classifier(models, parameters = parameteres)

#fitting the models on the training X and y
evaluator.fit(X_num_Train_c, y_train)

#estimating results from test dataset
evaluator.evaluate_and_score(X_num_Test_c,y_test)

#plotting the precision-recall curve in same graph for comparision
evaluator.plot_results()

#Outputing f1_score and auc for all models in a table
evaluator.results()

'''
evaluator.result constains all the information about all models in following schema:
{modelname:{'model',
            'best_fit_param',
            'cv_best_score',
            'f1_score',
            'auc' ,
            'precision',
            'recall'}}

'''


#Function to create correlation analysis. Output: "correlation of all columns with Y", "high intercorrelation column pairs" and "correlation matric of X"
def correlations(X,y,high_cor_thres = 0.6):

    col = list(X.columns)

    y_cor = []

    for i,x in np.transpose(X).iterrows():
        y_cor+= [(i,pearsonr(x,y)[0])]

    y_cor_output = pd.DataFrame(y_cor,columns = ['X_columns','y_cor'])


    cor_matrix = pd.DataFrame(X).corr()

    high_cor = []

    for i,x in enumerate(np.array(abs(cor_matrix)>high_cor_thres)):
        for c,y in enumerate(x):
            if y:
                if i>c:
                    high_cor+=[(col[i],col[c],cor_matrix.iloc[i,c])]

    high_cor_output = pd.DataFrame(high_cor,columns = ['col_name1','col_name2','cor'])
    
    return y_cor_output, high_cor_output, cor_matrix


correlations(X_num_Train_c,y_train)






