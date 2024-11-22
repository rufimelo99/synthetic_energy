from typing import Callable, Dict, List, Tuple

import pandas as pd
from faker import Faker

from synthetic_energy.anonymization.classifiers import classifier
from synthetic_energy.anonymization.schemas import AnonymizationType
from synthetic_energy.logger import logger


class Anonymizer:
    """
    Anonymizes sensitive data in a dataframe by applying various anonymization techniques.

    The class uses the Faker library to generate fake data such as names, emails, phone numbers,
    and other types of sensitive information. It can anonymize specific columns in a dataframe
    based on predefined anonymization types. The available anonymization techniques include
    generating fake names, emails, phone numbers, credit card details, and more.

    Parameters
    ----------
        anonymization_techniques : Dict[AnonymizationType, Callable]
            A dictionary mapping anonymization types to their respective anonymization methods.
        classifier : classifier.ColumnClassifier
            A classifier for determining column types for anonymization.

    """

    def __init__(self):
        """
        Initializes the anonymizer with a Faker instance and prepares a mapping of anonymization techniques
        to their corresponding methods.
        """
        self.faker = Faker()

        self.anonymization_techniques: Dict[AnonymizationType, Callable] = {
            AnonymizationType.NAME: self.anonymise_person_name,
            AnonymizationType.EMAIL: self.anonymise_email,
            AnonymizationType.PHONE_NUMBER: self.anonymise_phone_number,
            AnonymizationType.ADDRESS: self.anonymise_address,
            AnonymizationType.CREDIT_CARD_NUMBER: self.anonymise_credit_card_number,
            AnonymizationType.CREDIT_CARD_PROVIDER: self.anonymise_credit_card_provider,
            AnonymizationType.CREDIT_CARD_SECURITY_CODE: self.anonymise_credit_card_security_code,
            AnonymizationType.CREDIT_CARD_EXPIRATION_DATE: self.anonymise_credit_card_expiration_date,
            AnonymizationType.CREDIT_CARD_FULL: self.anonymise_credit_card_full,
            AnonymizationType.COMPANY: self.anonymise_company,
            AnonymizationType.SSN: self.anonymise_ssn,
            AnonymizationType.IPV4: self.anonymise_ipv4,
            AnonymizationType.IPV6: self.anonymise_ipv6,
            AnonymizationType.URL: self.anonymise_url,
            AnonymizationType.NON_SENSIBLE_DATA: lambda x, _: x,
            AnonymizationType.OTHER: self.anonymise_other,
        }

        self.classifier = classifier.ColumnClassifier()

    def anonymise_person_name(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing person names by replacing them with fake names.

        For each unique value in the specified column, a fake name is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing person names to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake names.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_names = {
            value: self.faker.name() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_names[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_email(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing email addresses by replacing them with fake email addresses.

        For each unique email in the specified column, a fake email address is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing email addresses to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake email addresses.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_emails = {
            value: self.faker.email() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_emails[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_phone_number(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing phone numbers by replacing them with fake phone numbers.

        For each unique phone number in the specified column, a fake phone number is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing phone numbers to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake phone numbers.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_phone_numbers = {
            value: self.faker.phone_number()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_phone_numbers[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_address(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing addresses by replacing them with fake addresses.

        For each unique address in the specified column, a fake address is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing addresses to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake addresses.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_addresses = {
            value: self.faker.address() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_addresses[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_credit_card_number(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing credit card numbers by replacing them with fake credit card numbers.

        For each unique credit card number in the specified column, a fake credit card number is generated using
        the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing credit card numbers to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake credit card numbers.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_credit_card_numbers = {
            value: self.faker.credit_card_number()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_credit_card_numbers[x] if pd.notnull(x) else x
        )

        return dataframe

    def anonymise_credit_card_provider(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing credit card providers by replacing them with fake credit card providers.

        For each unique credit card provider in the specified column, a fake provider is generated
        using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing credit card providers to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake credit card providers.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_credit_card_providers = {
            value: self.faker.credit_card_provider()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_credit_card_providers[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_credit_card_security_code(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing credit card security codes by replacing them with fake credit card security codes.

        For each unique security code in the specified column, a fake code is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing credit card security codes to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake credit card security codes.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_credit_card_security_codes = {
            value: self.faker.credit_card_security_code()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_credit_card_security_codes[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_credit_card_expiration_date(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing credit card expiration dates by replacing them with fake credit card expiration dates.

        For each unique expiration date in the specified column, a fake expiration date is generated
        using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing credit card expiration dates to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake credit card expiration dates.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_credit_card_expiration_dates = {
            value: self.faker.credit_card_expire()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: (
                unique_fake_credit_card_expiration_dates[x] if pd.notnull(x) else x
            )
        )
        return dataframe

    def anonymise_credit_card_full(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing full credit card details by replacing them with fake full credit card details.

        For each unique full credit card value in the specified column, fake credit card details (number, provider,
        expiration date, etc.) are generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing full credit card details to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake full credit card details.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_credit_card_full = {
            value: self.faker.credit_card_full()
            for value in unique_values
            if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_credit_card_full[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_company(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing company names by replacing them with fake company names.

        For each unique company name in the specified column, a fake company name is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing company names to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake company names.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_companies = {
            value: self.faker.company() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_companies[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_ssn(self, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Anonymizes a column containing social security numbers by replacing them with fake social security numbers.

        For each unique SSN in the specified column, a fake SSN is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing social security numbers to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake SSNs.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_ssns = {
            value: self.faker.ssn() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_ssns[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_ipv4(self, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Anonymizes a column containing IPv4 addresses by replacing them with fake IPv4 addresses.

        For each unique IPv4 address in the specified column, a fake address is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing IPv4 addresses to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake IPv4 addresses.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_ipv4s = {
            value: self.faker.ipv4() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_ipv4s[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_ipv6(self, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Anonymizes a column containing IPv6 addresses by replacing them with fake IPv6 addresses.

        For each unique IPv6 address in the specified column, a fake address is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing IPv6 addresses to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake IPv6 addresses.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_ipv6s = {
            value: self.faker.ipv6() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_ipv6s[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_url(self, dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Anonymizes a column containing URLs by replacing them with fake URLs.

        For each unique URL in the specified column, a fake URL is generated using the Faker library.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing URLs to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake URLs.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_urls = {
            value: self.faker.url() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: unique_fake_urls[x] if pd.notnull(x) else x
        )
        return dataframe

    def anonymise_other(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Anonymizes a column containing other types of data by replacing them with fake data.

        For each unique value in the specified column, a fake word is generated using the Faker library.
        The generated fake data will be prefixed with the column name to ensure uniqueness.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to anonymize.
        column_name : str
            The name of the column containing data to be anonymized.

        Returns
        -------
        pd.DataFrame
            The dataframe with the specified column anonymized with fake data.
        """
        unique_values = dataframe[column_name].unique()
        unique_fake_data = {
            value: self.faker.word() for value in unique_values if pd.notnull(value)
        }
        dataframe[column_name] = dataframe[column_name].apply(
            lambda x: column_name + unique_fake_data[x] if pd.notnull(x) else x
        )
        return dataframe

    def _map_anonymization_technique(self, technique: AnonymizationType) -> Callable:
        """
        Maps an anonymization technique to the corresponding anonymization function.

        This method returns the appropriate anonymization function for a given anonymization type.

        Parameters
        ----------
        technique : AnonymizationType
            The anonymization technique to be applied (e.g., NAME, EMAIL, SSN).

        Returns
        -------
        Callable
            The function corresponding to the given anonymization technique.
        """
        return self.anonymization_techniques[technique]

    def get_anonymisable_columns(self, dataframe: pd.DataFrame) -> list:
        """
        Returns a list of columns that contain personally identifiable information (PII).

        This method checks for columns that contain data types such as strings, which are more likely to contain PII, and returns a list of those columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to inspect for anonymizable columns.

        Returns
        -------
        list
            A list of column names containing potentially anonymizable data (i.e., columns of type 'object').
        """
        return dataframe.select_dtypes(include=["object"]).columns.tolist()

    def get_anonymization_technique(
        self, dataframe: pd.DataFrame, column_name: str
    ) -> AnonymizationType:
        """
        Returns the anonymization technique that should be used for a given column based on its contents.

        This method predicts the appropriate anonymization technique using a classifier by analyzing the first 10 values of the given column.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe containing the column to analyze.
        column_name : str
            The name of the column to assess.

        Returns
        -------
        AnonymizationType
            The anonymization technique (as an enum) to apply to the column.
        """
        values = dataframe[column_name].values[:10]
        return self.classifier.predict(values)

    def identify_sensible_columns(
        self, dataframe: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Identifies columns that contain sensitive information and determines the appropriate anonymization technique for each.

        This method uses the classifier to detect columns containing sensitive data and maps the appropriate anonymization techniques to them.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to analyze for sensitive columns.

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            A tuple containing:
            - A list of column names that contain sensitive information.
            - A dictionary mapping the sensitive columns to their corresponding anonymization techniques (as enum values).
        """
        sensible_columns = []
        anonymisation_mappings = {}
        logger.debug("Dtype of columns", dtypes=dataframe.dtypes)
        anonymisable_columns = self.get_anonymisable_columns(dataframe)
        logger.debug("Anonymisable columns", columns=anonymisable_columns)

        for column in anonymisable_columns:
            technique = self.get_anonymization_technique(dataframe, column)

            logger.debug(
                "Associated anonymization technique", technique=technique, column=column
            )

            if technique != AnonymizationType.NON_SENSIBLE_DATA:
                sensible_columns.append(column)
                anonymisation_mappings[column] = technique.value

        return sensible_columns, anonymisation_mappings

    def anonymize(
        self, dataframe: pd.DataFrame, sensitive_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Anonymizes the data by replacing personally identifiable information (PII) with fake data.

        For each column that is identified as containing sensitive information, this method will apply the appropriate anonymization technique.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to anonymize.
        sensitive_columns : Dict[str, str]
            A dictionary mapping column names to their corresponding anonymization technique (as string values).

        Returns
        -------
        pd.DataFrame
            The dataframe with sensitive data anonymized.
        """
        for column, technique in sensitive_columns.items():
            technique = AnonymizationType(technique)
            anonymization_function = self._map_anonymization_technique(technique)
            dataframe = anonymization_function(dataframe, column)

        return dataframe
