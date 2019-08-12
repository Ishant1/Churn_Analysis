
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,FeatureUnion




class type_setter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.record = {}


    def fit(self,X,y=None,date=[]):

        for x in X:
            #if unique values are less than 20 then factor
            if x in date:
                self.record[x] = 'datetime64'
            else:
                if X[x].nunique()>20:
                    try:
                        X[x].mean()
                        self.record[x] = 'float'
                     
                    except:
                        self.record[x] = 'object'
                        print("erro with {}".format(x))
                
                else:
                    self.record[x] = 'category'
  
        return self

    def transform(self,X,y=None):
        X_c = X.copy()
        for x,i in self.record.items():
            X_c[x] = X_c[x].astype(i)
        return X_c



class select_type(BaseEstimator,TransformerMixin):
    def __init__(self,type = 'object'):
        self.type = type

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X.select_dtypes(self.type)

class imputer(SimpleImputer,BaseEstimator, TransformerMixin):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def transform(self,X,y=None):
        copy = X.copy()
        copy.loc[:,:] = super(imputer,self).transform(copy)
        return copy

    


class Standardizer(StandardScaler,BaseEstimator, TransformerMixin):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,X,y=None):
        super(Standardizer,self).fit(X)
        self.stats_ = pd.DataFrame({'mean':X.select_dtypes('number').mean().to_dict(),
                                    'std':X.select_dtypes('number').std().to_dict()})
        return self

    def transform(self,X,y=None):
        copy = X.copy()
        copy.loc[:,:] = super(Standardizer,self).transform(copy)
        return copy



class Encoder(OneHotEncoder,BaseEstimator, TransformerMixin):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)

        def fit(self, X, y=None):
            super().fit(X)
            #self.columns_ = list(pd.get_dummies(X,drop_first = True))
            return self

        def transform(self, X):
            return pd.DataFrame(super().transform(X).toarray(),columns = super().get_feature_names())





#----
def null_fill(Data_,type = 'char'):
    copy = Data_.copy()
    copy = copy.replace( ['null','NA','Null'],np.nan)
    if any(copy.isnull().values):
        if type == 'char':
            copy = copy.fillna(copy.mode()[0])
        elif type =='num':
            copy = copy.fillna(copy.mean())
    return copy