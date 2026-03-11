from zenml import pipeline, step
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
@step
def data_loader() -> pd.DataFrame:
    # step 1 : load DVC - tracked data
    df = pd.read_csv("/workspaces/smart-real-estate-predictor/archive/AmesHousing.csv")
    return df
# step 2 basic cleaning
@step
def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop columns with too many missing values (e.g., Pool quality)
    # This is better than dropping all rows
    df = df.drop(columns=['Pool QC', 'Misc Feature', 'Alley', 'Fence'])

    # 2. Fill missing numeric values with the median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 3. Handle Categorical data
    # Random Forest needs numbers, not strings. 
    # For now, let's keep it simple and only use numeric columns for training.
    df = df.select_dtypes(include=['number'])

    return df
@step(experiment_tracker="mlflow_tracker")
def model_trained(df:pd.DataFrame):
    #step3 split and train
    X =  df.drop("SalePrice", axis = 1)
    y = df["SalePrice"]
    X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train ,  y_train)
    sc=model.score(X_test ,  y_test)
    print(sc * 100)
    #automatically logging parameters and metrics to the MLFlow UI
    mlflow.sklearn.autolog()
    #evaluation
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"Model MSE : {mse}")
    return model
@pipeline
def smart_real_estate_pipeline():
    #glueing all the steps together
    dataset = data_loader()
    clean_data = preprocessing(dataset)
    model = model_trained(clean_data)