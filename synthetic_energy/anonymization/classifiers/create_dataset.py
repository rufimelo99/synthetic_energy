"""
For the anonymization of the data, we will use classifiers to determine the anonymization technique that should be used for each column. 
For now, we will create 1000 entries of fake data for each column that contains personally identifiable information.
We will use the Faker library to generate fake data.
# TODO(Rui/Andre): Ideally, in the future, we want to scrappe and compile data from multiple sources to train the classifiers.
"""

import random

import pandas as pd
from faker import Faker
from tqdm import tqdm

from synthetic_energy.logger import logger

faker = Faker()

N_INDIVIDUAL_ENTRIES = 100


def preprocess_data(text: str) -> str:
    """
    Preprocesses the text data.
    """
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")


def create_dataset() -> pd.DataFrame:
    """
    Creates a dataset with fake data.
    """

    columns = ["data", "label"]

    tqdm.write("Creating dataset...")

    dataframe = pd.DataFrame(columns=columns)

    tqdm.write("Creating fake person names...")

    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.name()
        data = preprocess_data(data)
        # use concat to add the data to the dataframe
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["name"]})],
            ignore_index=True,
        )

    tqdm.write("Creating fake emails...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.email()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["email"]})],
            ignore_index=True,
        )

    tqdm.write("Creating fake phone numbers...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.phone_number()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["phone_number"]})],
            ignore_index=True,
        )

    tqdm.write("Creating fake addresses...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.address()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["address"]})],
            ignore_index=True,
        )

    tqdm.write("Creating fake credit card numbers...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.credit_card_number()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [
                dataframe,
                pd.DataFrame({"data": [data], "label": ["credit_card_number"]}),
            ],
            ignore_index=True,
        )

    tqdm.write("Creating fake URLs...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.url()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["url"]})],
            ignore_index=True,
        )

    tqdm.write("Creating fake Company names...")
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data = faker.company()
        data = preprocess_data(data)
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["company"]})],
            ignore_index=True,
        )

    tqdm.write("Non-sensible-data generation")
    data_generators = {
        "text": lambda: faker.text(max_nb_chars=20),
        "barcode": lambda: random.choice([faker.isbn13(), faker.ean13(), faker.ean8()]),
        "date": lambda: faker.date(),
        "number": lambda: faker.random_number(digits=5),
        "misc": lambda: random.choice(
            [faker.currency_code(), faker.color_name(), faker.file_name()]
        ),
    }
    for _ in tqdm(range(N_INDIVIDUAL_ENTRIES)):
        data_type = random.choice(["text", "barcode", "date", "number", "misc"])

        data = data_generators[data_type]()

        data = str(data)
        data = preprocess_data(data) if not data.isnumeric() else data
        dataframe = pd.concat(
            [dataframe, pd.DataFrame({"data": [data], "label": ["non_sensible_data"]})],
            ignore_index=True,
        )

    return dataframe


if __name__ == "__main__":
    logger.info("Creating dataset...")
    dataset = create_dataset()
    dataset.to_csv("dataset.csv", index=False)
