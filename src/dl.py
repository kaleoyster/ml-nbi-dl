import numpy as np
import pandas as pd
import shap
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

import mlflow
import mlflow.tensorflow
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import clone_model

from deep_processing import *

def wide_and_deep_model(X_train):
    """
    Description:
        Performs the modeling (Deep learning) and returns performance metrics
    Args:
        X_train: Features of Training Set
    Return:
        model: Deep learning machine model
    """
    # Input layer
    input_layer = Input(shape=(X_train.shape[1],))

    # Wide part (Linear model)
    wide_layer = Dense(8, activation='relu')(input_layer)

    # Deep part (Deep Neural Network)
    deep_layer = Dense(64, activation='relu')(input_layer)
    deep_layer = Dropout(0.5)(deep_layer)
    deep_layer = Dense(32, activation='relu')(deep_layer)
    deep_layer = Dropout(0.5)(deep_layer)

    # Combine Wide and Deep parts
    combined_layers = Concatenate()([wide_layer, deep_layer])
    output_layer = Dense(2, activation='softmax')(combined_layers)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define a function for preprocessing and training the model
def preprocess_and_train(model, X_train, y_train, X_test, y_test):
    # Sampling
    sampling = SMOTE()
    X_train, y_train = sampling.fit_resample(X_train, y_train)

    # Label encoding
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_train = pd.get_dummies(y_train).values
    y_test = encoder.fit_transform(y_test)
    y_test = pd.get_dummies(y_test).values

    # Convert to arrays
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Model initialization and training
    model.fit(X_train, y_train, batch_size=32, epochs=500)

    # Example: Calculate and print metrics for each fold
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    roc = roc_auc_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), multi_class='ovr')
    kappa = cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    print(f'AUC: {auc}, ROC: {roc}, Kappa: {kappa}, F1: {f1}, Accuracy: {accuracy}')
    return model, auc, roc, kappa, f1, accuracy, (y_pred, y_test)

def main():
    # Run name
    model_name =  'deep-learning-nbi'
    sampling_technique = 'smote'
    holdout_set_size = '0'

    # Preprocess dataset
    bridge_X, bridge_y, holdout, cols = preprocess()

    # K-fold cross-validation
    kFold = KFold(n_splits=2, shuffle=True, random_state=1)

    # Model
    model = wide_and_deep_model(bridge_X)

    # Parameters
    with open("parameters.yaml", "r") as file:
        parameter = yaml.safe_load(file)

    # Generate a unique experiment name with a timestamp
    experiment_name = f"{model_name}-{datetime.now().strftime('%Y%m%d')}"
    run_name = model_name + \
        '- ' + \
        sampling_technique + \
        '-' + \
        holdout_set_size + \
        '-' + \
        str(parameter['test_size'])

    ## Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create a new experiment if it does not exist
        experiment_id = mlflow.create_experiment((experiment_name))
    else:
        # If it exists, get the experiment ID
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Initialize lists to store metric values for each fold
    auc_scores = []
    roc_scores = []
    kappa_scores = []
    f1_scores = []
    accuracy_scores = []

    with mlflow.start_run(run_name=run_name) as run:
        #Log metrics
        mlflow.tensorflow.autolog()
         # Perform K-fold cross-validation
        for foldTrainX, foldTestX in kFold.split(bridge_X):
            X_train, y_train, X_test, y_test = bridge_X[foldTrainX], bridge_y[foldTrainX], \
                                                bridge_X[foldTestX], bridge_y[foldTestX]

            model, auc, roc, kappa, f1, acc, class_set = preprocess_and_train(model, X_train, y_train, X_test, y_test)
            model = clone_model(model)
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Append metric values to lists
            auc_scores.append(auc)
            roc_scores.append(roc)
            kappa_scores.append(kappa)
            f1_scores.append(f1)
            accuracy_scores.append(acc)

        mean_auc = np.mean(auc_scores)
        mean_roc = np.mean(roc_scores)
        mean_kappa = np.mean(kappa_scores)
        mean_f1 = np.mean(f1_scores)
        mean_acc = np.mean(accuracy_scores)

        print(f'AUC: {mean_auc}, ROC: {mean_roc}, Kappa: {mean_kappa}, F1: {mean_f1}, Accuracy: {mean_acc}')

        #X_test_new = pd.DataFrame(X_test, columns=cols)
        mlflow.log_metric("accuracy", mean_acc)
        mlflow.log_metric("auc", mean_acc)
        mlflow.log_metric("f1_score", mean_f1)
        mlflow.log_metric("kappa", mean_kappa)
        
        # Compute SHAP values using a subset of the test data for efficiency
        #X_train_new = pd.DataFrame(X_train, columns=cols)
        #X_test_new = pd.DataFrame(X_test, columns=cols)

        ## Compute SHAP Values
        X_train = X_train.astype('float32')

        #explainer = shap.Explainer(model, bridge_X[:10])
        #shap_values = explainer(bridge_X[:10])
        #mean_shap = np.mean(abs(shap_values.values), axis=0).mean(1)
        #mean_shap_features = {column:shap_v for column, shap_v in zip(cols, mean_shap)}

        explainer = shap.DeepExplainer(model, X_train[:100])
        shap_values = explainer.shap_values(X_train[:10])

        # Log SHAP values as an artifact (you can save them as a plot or in a file)
        # For simplicity, we'll save the mean absolute SHAP values per feature as a CSV
        shap_sum = np.abs(shap_values[0]).mean(axis=0)
        np.savetxt("shap_values.csv", shap_sum, delimiter=",")
        mlflow.log_artifact("shap_values.csv", "shap_values")

        ## If you want to log a SHAP summary plot as an image
        shap.summary_plot(shap_values, X_train[:10], show=False)
        plt.savefig('shap_summary_plot.png')
        mlflow.log_artifact('shap_summary_plot.png', 'shap_plots')

    # Log metrics with MLflow
    #y_test_new, y_pred_new = class_set
    #tp, tn, fn, fp = get_classification_samples(X_test, y_test_new, y_pred_new, cols, model_name)
    #tp.to_csv("true_positives.csv", index=False)
    #tn.to_csv("true_negatives.csv", index=False)
    #fn.to_csv("false_negatives.csv", index=False)
    #fp.to_csv("false_positives.csv", index=False)

    # List artifacts
    #client = MlflowClient()
    #artifact_path = "model_explanations_shap"
    #artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
    #
    #print("# artifacts:")
    #print(artifacts)
    #
    ### load back the logged explanation
    #dst_path = download_artifacts(run_id=run.info.run_id, artifact_path=artifact_path)
    #base_values = np.load(os.path.join(dst_path, "base_values.npy"))
    #shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

if __name__=='__main__':
    main()
