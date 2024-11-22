"""
    We will train a multi-class classifier to determine the anonymization technique that should be used for each column.
"""

import pickle
from argparse import ArgumentParser

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from synthetic_energy.anonymization.classifiers.classifier import FeatureEngineering
from synthetic_energy.logger import logger


def feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the dataset.
    """
    new_dataframe = pd.DataFrame()

    # Pipeline for feature engineering
    fe = FeatureEngineering()

    for column in dataframe.columns:
        data = dataframe[column].values
        features, feature_names = fe.pipeline(data, get_features_names=True)
        for feature in features:
            new_dataframe = pd.concat(
                [
                    new_dataframe,
                    pd.DataFrame(
                        {
                            feature_names[j]: [feature[j]]
                            for j in range(len(feature_names))
                        }
                    ),
                ],
                ignore_index=True,
            )
    # Drop nan values
    new_dataframe["label"] = dataframe["label"]
    new_dataframe["data"] = dataframe["data"]
    new_dataframe = new_dataframe.dropna()
    return new_dataframe


def train_classifier(pd_dataframe: pd.DataFrame):
    """
    Trains a classifier to determine the anonymization technique that should be used for each column.
    """
    # Convert the dataframe into a numpy array
    y = pd_dataframe["label"].values
    X = pd_dataframe.drop(columns=["label", "data"]).values

    # Sfuffle the data
    X, y = shuffle(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    labeler_encoder = LabelEncoder()
    y_train = labeler_encoder.fit_transform(y_train)
    y_test = labeler_encoder.transform(y_test)

    # Train the classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    score = classifier.score(X_test, y_test)
    logger.info("Evaluting classifier...", score=score)

    return classifier, labeler_encoder


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", type=str, help="Path to the dataset")
    args = parser.parse_args()

    base_dataframe = pd.read_csv(args.data)
    dataframe = feature_engineering(base_dataframe)

    dataframe.to_csv("data.csv", index=False)
    if base_dataframe.shape[0] != dataframe.shape[0]:
        logger.error("Some entries were removed or added during feature engineering.")
        raise ValueError(
            "Some entries were removed or added during feature engineering."
        )

    classifier, labeler_encoder = train_classifier(dataframe)
    with open("multi_column_classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)

    with open("labeler_encoder.pkl", "wb") as f:
        pickle.dump(labeler_encoder, f)
