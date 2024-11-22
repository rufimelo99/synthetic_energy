import os
import pickle
import re
from collections import Counter
from typing import List, Union

import numpy as np
import spacy

from synthetic_energy.anonymization.schemas import AnonymizationType
from synthetic_energy.logger import logger

NER_MODEL = spacy.load("en_core_web_sm")

# Correctly define DEFAULT_MODEL_PATH at the global level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "multi_column_classifier.pkl")
DEFAULT_LABELER_ENCODER_PATH = os.path.join(BASE_DIR, "labeler_encoder.pkl")


def ends_in_com_org_net(x: str) -> bool:
    # Use regex to check if the string ends in .com, .org, .net, .edu, .gov, .pt, .en, .uk.
    # TODO(Rui/Andre): Add more TLDs.
    has_com_org_net = (
        bool(re.search(r"\.com$", x))
        or bool(re.search(r"\.org$", x))
        or bool(re.search(r"\.net$", x))
        or bool(re.search(r"\.edu$", x))
        or bool(re.search(r"\.gov$", x))
        or bool(re.search(r"\.pt$", x))
        or bool(re.search(r"\.en$", x))
        or bool(re.search(r"\.uk$", x))
    )
    return 1 if has_com_org_net else 0


def has_at(x: str) -> int:
    return 1 if "@" in x else 0


def has_any_letters(x: str) -> bool:
    return 1 if [char for char in x if char.isalpha()] else 0


def has_any_numbers(x: str) -> bool:
    return 1 if [char for char in x if char.isdigit()] else 0


def has_special_characters(x: str) -> bool:
    return 1 if x.isalnum() else 0


def has_spaces(x: str) -> bool:
    return 1 if " " in x else 0


def has_punctuation(x: str) -> bool:
    return 1 if "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" in x else 0


def is_capitalized(x: str) -> bool:
    return 1 if x[0].isupper() else 0


def has_http_https_www(x: str) -> bool:
    return 1 if "http" in x or "https" in x or "www" in x else 0


def has_com_org_net(x: str) -> bool:
    return ends_in_com_org_net(x)


def has_person_name_from_ner(x: str) -> bool:
    entities = NER_MODEL(x).ents
    if entities:
        for entity in entities:
            if entity.label_ == "PERSON":
                return 1
    return 0


def has_city_name_from_ner(x: str) -> bool:
    entities = NER_MODEL(x).ents
    if entities:
        for entity in entities:
            if entity.label_ == "GPE":
                return 1
    return 0


def has_organization_from_ner(x: str) -> bool:
    entities = NER_MODEL(x).ents
    if entities:
        for entity in entities:
            if entity.label_ == "ORG":
                return 1
    return 0


class FeatureEngineering:
    @staticmethod
    def pipeline(text: Union[str, List[str]], get_features_names=False):
        """
        Pipeline for feature engineering.
        """
        if isinstance(text, str):
            text = [text]

        # tokenizer from an LLM to create features ?
        # Count @
        # Count :
        # Count /
        # Length of the string
        # has letters (phone numbers
        # has numbers
        # has special characters
        # has spaces
        # has punctuation
        # has uppercase
        # has http, https or www
        # has .com, .org, .net, .edu, .gov, .pt, .en, .uk, ... -> Use regex

        features = {
            "has_@": has_at,
            "has_any_letters": has_any_letters,
            "has_any_numbers": has_any_numbers,
            "has_special_characters": has_special_characters,
            "has_spaces": has_spaces,
            "has_punctuation": has_punctuation,
            "is_capitalized": is_capitalized,
            "has_http_https_www": has_http_https_www,
            "has_com_org_net": has_com_org_net,
            "has_person_name": has_person_name_from_ner,
            "has_city_name": has_city_name_from_ner,
            "has_organization": has_organization_from_ner,
            # has lots of numbers
            # counter "."
        }

        data = []
        for entry in text:
            entry = str(entry)
            data_point_features = []
            for feature in features.values():
                data_point_features.append(feature(entry))
            data.append(data_point_features)
        if get_features_names:
            return data, list(features.keys())

        return data


SUPPORTED_TYPES = Union[str, List[str], float, List[float], int, List[int]]


class ColumnClassifier:
    """
    Column classifier classifies columns based on their data to understand what kind of anonymization technique should be used.
    """

    def __init__(self):
        with open(DEFAULT_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(DEFAULT_LABELER_ENCODER_PATH, "rb") as f:
            self.labeler_encoder = pickle.load(f)

    def predict(self, data: SUPPORTED_TYPES) -> AnonymizationType:
        """
        Predicts the anonymization type for each column.
        """
        if not isinstance(data, list):
            data = [data]

        for element in data:
            element = str(element)

        features = FeatureEngineering.pipeline(data)
        predict_proba = self.model.predict_proba(features).tolist()

        predictions = []
        for idx, prediction in enumerate(predict_proba):
            logger.debug("Probabilities from classifier", probabilities=prediction)
            if (
                max(prediction) < (1 / len(self.labeler_encoder.classes_)) * 2
            ):  # TODO(Rui): This is a hack.
                prediction = AnonymizationType.NON_SENSIBLE_DATA
            else:
                logger.debug("Predicting entry from column", entry=data[idx])
                label = np.argmax(prediction).tolist()
                prediction = self.labeler_encoder.inverse_transform([label])
                if prediction:
                    prediction = AnonymizationType(prediction)
                else:
                    logger.warning(
                        "The model was not able to predict the anonymization type."
                    )
                    prediction = AnonymizationType.NON_SENSIBLE_DATA

            predictions.append(prediction)
        counter = Counter(predictions)
        prediction = counter.most_common(1)[0][0]
        return prediction
