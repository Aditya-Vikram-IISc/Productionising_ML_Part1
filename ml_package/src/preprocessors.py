import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    """Drops the features from the columns
    """
    def __init__(self, variables: list | str = None):
        if not isinstance(variables, list):
            self.variables = [variables]

        self.variables = variables

    def fit(self, X:pd.DataFrame, y:pd.Series= None)-> 'DropFeatures':
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X
    

class NumericalImputer(BaseEstimator, TransformerMixin):
    """Imputes Numerical Column basis the mean or mode as specified
    """
    def __init__(self, variables: list| str = None, mode: str =None):
        if not isinstance(variables, list):
            self.variables = variables
        self.variables = variables
        self.mode = mode

    def fit(self, X:pd.DataFrame, y:pd.Series = None)-> 'NumericalImputer':
        self.imputer_dict = {}
        if self.mode == "mean":
            for feature in self.variables:
                self.imputer_dict[feature] = X[feature].mean()

        elif self.mode == "mode":
            for feature in self.variables:
                self.imputer_dict[feature] = X[feature].mode()[0]

        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.imputer_dict[feature])
        
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables: list|str = None, mode:str = None)-> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        self.variables = variables
        self.mode = mode

    def fit(self, X:pd.DataFrame, y:pd.Series=None)-> 'CategoricalImputer':
        self.imputer_dict = {}
        if self.mode == "missing":
            for feature in self.variables:
                if feature in X.columns:
                    self.imputer_dict[feature] = "missing"
                else:
                    raise ValueError(f"Feature {feature} not in Dataframe")
                

        elif self.mode == "mode":
            for feature in self.variables:
                if feature in X.columns:
                    self.imputer_dict[feature] = X[feature].mode()[0]
                else:
                    raise ValueError(f"Feature {feature} not forund in DataFrame")

                

        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()

        for feature in self.variables:
            if feature in X.columns:
                X[feature] = X[feature].fillna(self.imputer_dict[feature])

            else:
                raise ValueError(f"Feature {feature} not forund in DataFrame")



class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables: list|str = None, mode:str =None):
        if not isinstance(variables, list):
            self.variables = [variables]
        self.variables = variables
        self.mode = mode

    def fit(self, X:pd.DataFrame, y:pd.Series= None)-> 'CategoricalEncoder':
        self.encoder_dict = {}

        if self.mode == "by_countorder":
        
            for variable in self.variables:
                t = X[variable].value_counts().sort_values(ascending=False).index
                self.encoder_dict[variable] = {k:i for i,k in enumerate(t, start=0)}
        # write for custom as well
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for variable in self.variables:
            if variable in X.columns:
                X[variable]  = X[variable].map(self.encoder_dict[variable])
            else:
                raise ValueError(f"Feature {variable} not predent in DataFrame Columns")
