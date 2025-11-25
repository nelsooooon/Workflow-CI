import mlflow
import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    np.random.seed(40)

    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv")

    x = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities for confidence scoring
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
            )
        
        accuracy = model.score(X_test, y_test)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("avg_confidence_class_0", y_pred_proba[:, 0].mean())
        mlflow.log_metric("avg_confidence_class_1", y_pred_proba[:, 1].mean())
