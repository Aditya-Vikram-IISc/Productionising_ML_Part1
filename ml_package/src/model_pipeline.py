from .preprocessors import DropFeatures, NumericalImputer, CategoricalImputer, CategoricalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer



class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 features_to_drop =None,
                 numerical_columns = None,
                 numerical_column_impute_strategy = None,
                 categorical_column = None,
                 categorical_column_impute_strategy = None,
                 categorical_column_encode_strategy = None
                 ):
        
        self.features_to_drop = features_to_drop
        self.numerical_columns = numerical_columns
        self.numerical_column_impute_strategy = numerical_column_impute_strategy
        self.categorical_column = categorical_column
        self.categorical_column_impute_strategy = categorical_column_impute_strategy
        self.categorical_column_encode_strategy = categorical_column_encode_strategy
        self.pipeline = None

    def _build_pipeline(self):
        mixmax_Scaler = ColumnTransformer(
            transformers=[('scaler', MinMaxScaler(), self.numerical_columns)],
            remainder= "passthrough"
        )

        # Define the full pipeline
        pipeline = Pipeline([
                    ("drop_features", DropFeatures(variables = self.features_to_drop)),
                    ("catcol_imputer", CategoricalImputer(variables = self.categorical_column, mode = self.categorical_column_impute_strategy)),
                    ("catcol_encoder", CategoricalEncoder(variables = self.categorical_column, mode = self.categorical_column_encode_strategy)),
                    ("numcol_imputer", NumericalImputer(variables = self.numerical_columns)),
                    ("numcol_scaler", mixmax_Scaler)

        ])

        self.pipeline = pipeline

    def fit(self, X, y=None):
        self._build_pipeline()
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        self._build_pipeline()
        return self.pipeline.fit_transform(X, y)

