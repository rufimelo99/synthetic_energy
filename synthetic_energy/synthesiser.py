from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from visions.functional import cast_to_inferred
from visions.typesets import StandardSet

from synthetic_energy import configs, schemas
from synthetic_energy.anonymization import anonymizer
from synthetic_energy.logger import logger
from synthetic_energy.quality_metrics.kl_divergence import PopulationStabilityIndex
from synthetic_energy.tabular.actgan import actgan
from synthetic_energy.tabular.actgan.structures import ConditionalVectorType
from synthetic_energy.time_series.doppelganger.config import DGANConfig
from synthetic_energy.time_series.doppelganger.doppelganger import DGAN
from synthetic_energy.utils import has_datetime_columns, is_time_series

TYPES_TO_NORMALISE = {
    "int64",
    "float64",
}  # TODO(Rui): add more types! and create TYPES_TO_NORMALISE in config.


class NormalisationInfoHolder:
    """
    Handles feature normalization and denormalization for numerical data.

    This class applies different normalization techniques (min-max scaling, z-score,
    or robust scaling) to numerical features of a dataframe and stores the necessary
    information for inverse normalization.

    Attributes
    ----------
    normalisation_info : Dict[str, Tuple[float, float]]
        Stores normalization parameters for each column. The parameters vary
        based on the normalization technique applied.

    Methods
    -------
    normalise(dataframe: pd.DataFrame, config: configs.DataProcessingConfig) -> pd.DataFrame
        Apply the specified normalization technique to numerical features in the dataframe.
    inverse_normalisation(dataframe: pd.DataFrame) -> pd.DataFrame
        Revert the normalization process and return the data to its original scale.
    """

    def __init__(self):
        """
        Initialize an empty normalization info holder.
        """
        self.normalisation_info = {}

    def __add_normalisation_info(self, column, min_val, max_val):
        """
        Store normalization parameters for a specific column.

        Parameters
        ----------
        column : str
            The name of the column being normalized.
        min_val : float
            The minimum value (or mean/median depending on the normalization type) of the column.
        max_val : float
            The maximum value (or standard deviation/IQR depending on the normalization type) of the column.
        """
        self.normalisation_info[column] = (min_val, max_val)

    def normalise(
        self, dataframe: pd.DataFrame, config: configs.DataProcessingConfig
    ) -> pd.DataFrame:
        """
        Normalize numerical features in the dataframe based on the specified technique.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe containing features to normalize.
        config : configs.DataProcessingConfig
            The configuration specifying the normalization technique.

        Returns
        -------
        pd.DataFrame
            The normalized dataframe.

        Raises
        ------
        NotImplementedError
            If an unsupported normalization technique is specified in the configuration.
        """
        if config.normalisation == configs.NORMALISATION.MIN_MAX:
            for column in dataframe.columns:
                if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                    continue
                self.__add_normalisation_info(
                    column, dataframe[column].min(), dataframe[column].max()
                )
                dataframe[column] = (dataframe[column] - dataframe[column].min()) / (
                    dataframe[column].max() - dataframe[column].min()
                )
        elif config.normalisation == configs.NORMALISATION.Z_SCORE:
            for column in dataframe.columns:
                if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                    continue
                self.__add_normalisation_info(
                    column, dataframe[column].mean(), dataframe[column].std()
                )
                dataframe[column] = (
                    dataframe[column] - dataframe[column].mean()
                ) / dataframe[column].std()
        elif config.normalisation == configs.NORMALISATION.ROBUST:
            for column in dataframe.columns:
                if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                    continue
                self.__add_normalisation_info(
                    column,
                    dataframe[column].median(),
                    dataframe[column].quantile(0.75) - dataframe[column].quantile(0.25),
                )
                dataframe[column] = (dataframe[column] - dataframe[column].median()) / (
                    dataframe[column].quantile(0.75) - dataframe[column].quantile(0.25)
                )
        else:
            raise NotImplementedError
        return dataframe

    def inverse_normalisation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Revert the normalization process for the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The normalized dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe reverted to its original scale.
        """
        for column in dataframe.columns:
            if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                continue
            min_val, max_val = self.normalisation_info[column]
            dataframe[column] = dataframe[column] * (max_val - min_val) + min_val
        return dataframe


class CategoricalEncoder:
    """
    Handles label encoding and decoding for categorical features.

    This class applies label encoding to categorical features in a dataframe
    and provides methods for inverse transformation.

    Attributes
    ----------
    labeler_encoders : Dict[str, LabelEncoder]
        Stores label encoders for each categorical feature.
    categorical_features : List[str]
        List of categorical feature names in the dataframe.
    config : configs.DataProcessingConfig
        The configuration specifying the categorical encoding method.

    Methods
    -------
    label_encode(dataframe: pd.DataFrame, config: configs.DataProcessingConfig) -> pd.DataFrame
        Encode categorical features in the dataframe using label encoding.
    reverse_label_encode(dataframe: pd.DataFrame) -> pd.DataFrame
        Decode previously encoded categorical features back to their original values.
    """

    def __init__(
        self, config: configs.DataProcessingConfig = configs.DataProcessingConfig()
    ):
        """
        Initialize the encoder with a given configuration.

        Parameters
        ----------
        config : configs.DataProcessingConfig, optional
            Configuration specifying the categorical encoding strategy (default: LABEL encoding).
        """
        self.labeler_encoders = {}
        self.categorical_features = []
        self.config = config

    def __add_labeler_encoder(self, feature, label_encoder):
        """
        Store the label encoder for a specific feature.

        Parameters
        ----------
        feature : str
            The name of the feature.
        label_encoder : LabelEncoder
            The label encoder instance used for the feature.
        """
        self.labeler_encoders[feature] = label_encoder

    def label_encode(
        self, dataframe: pd.DataFrame, config: configs.DataProcessingConfig
    ) -> pd.DataFrame:
        """
        Encode categorical features in the dataframe using label encoding.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe containing categorical features.
        config : configs.DataProcessingConfig
            The configuration specifying the categorical encoding strategy.

        Returns
        -------
        pd.DataFrame
            The dataframe with encoded categorical features.

        Raises
        ------
        NotImplementedError
            If an unsupported categorical encoding strategy is specified.
        """
        if self.config.categorical_encoding == configs.CATEGORICAL_ENCODING.NONE:
            return dataframe
        elif self.config.categorical_encoding == configs.CATEGORICAL_ENCODING.LABEL:
            for feature in dataframe.columns:
                if dataframe[feature].dtype == "object":
                    logger.info("Label encoding feature", feature=feature)
                    label_encoder = LabelEncoder()
                    dataframe[feature] = label_encoder.fit_transform(dataframe[feature])
                    self.__add_labeler_encoder(feature, label_encoder)
            return dataframe
        else:
            raise NotImplementedError

    def reverse_label_encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Decode previously encoded categorical features back to their original values.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe with encoded categorical features.

        Returns
        -------
        pd.DataFrame
            The dataframe with decoded categorical features.
        """
        for feature in dataframe.columns:
            if feature in self.labeler_encoders:
                logger.info("Inverse label encoding feature", feature=feature)
                label_encoder: LabelEncoder = self.labeler_encoders[feature]
                new_values = np.round(dataframe[feature]).astype(int)
                res = []

                known_classes = label_encoder.classes_
                known_labels = label_encoder.transform(known_classes)

                for new_value in new_values:
                    if new_value in known_labels:
                        res.append(label_encoder.inverse_transform([new_value])[0])
                    else:
                        res.append("Unknown")
                dataframe[feature] = res
        return dataframe


class Synthesiser:
    """
    A class to preprocess, synthesize, and anonymize data for both tabular and time-series formats.

    Attributes
    ----------
    anonymizer : anonymizer.Anonymizer
        An instance of the Anonymizer class for anonymizing sensitive columns.

    Methods
    -------
    prepare_data(dataframe, config, normalisation_holder, categorical_encoder)
        Preprocesses the input dataframe by normalizing, encoding, and imputing missing values.
    associate_dtypes(dataframe, dtypes)
        Reapplies inferred data types to the dataframe.
    synthesise_data(dataframe, features, sensitive_columns, preprocessing_config, synthesiser_config)
        Prepares, synthesizes, and anonymizes data, returning the anonymized dataframe and a quality metric (PSI).
    synthesise_tabular_data(prepared_dataframe, features, normalisation_holder, categorical_encoder, synthesiser_config)
        Generates synthetic tabular data using the ACTGAN model.
    synthesise_time_series_data(prepared_dataframe, features, normalisation_holder, categorical_encoder, synthesiser_config)
        Generates synthetic time-series data using the DGAN model.
    identify_sensible_columns(dataframe)
        Identifies and returns sensitive columns in the dataframe.
    """

    def __init__(self):
        """
        Initializes the Synthesiser with an anonymizer.
        """
        self.anonymizer = anonymizer.Anonymizer()

    def prepare_data(
        self,
        dataframe: pd.DataFrame,
        config: configs.DataProcessingConfig,
        normalisation_holder: NormalisationInfoHolder,
        categorical_encoder: CategoricalEncoder,
    ) -> Tuple[pd.DataFrame, NormalisationInfoHolder, CategoricalEncoder, dict]:
        """
        Prepares the input dataframe for synthesis by performing normalization,
        categorical encoding, and missing value imputation.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe to be preprocessed.
        config : configs.DataProcessingConfig
            Configuration for data preprocessing, including normalization, imputation, and deduplication.
        normalisation_holder : NormalisationInfoHolder
            Holds normalization parameters and applies normalization.
        categorical_encoder : CategoricalEncoder
            Encodes and decodes categorical features.

        Returns
        -------
        Tuple[pd.DataFrame, NormalisationInfoHolder, CategoricalEncoder, dict]
            - Preprocessed dataframe.
            - Updated NormalisationInfoHolder instance.
            - Updated CategoricalEncoder instance.
            - Inferred types of the columns in the dataframe.
        """
        # If there is an index, create a new column with the index values.
        if dataframe.index.name:
            dataframe.reset_index(inplace=True)
            dataframe.rename(columns={dataframe.index.name: "df__index"}, inplace=True)

        # TODO(Rui): This might need to be moved elsewhere.
        typeset = StandardSet()
        dataframe = dataframe.astype(str)

        # Fill missing values with np.nan.
        dataframe = dataframe.fillna(np.nan)

        # TODO(Rui): We probably want to tackle each column individually to boost performance.
        dataframe = cast_to_inferred(dataframe, typeset)

        # Sees if we have timestamp columns.
        infered_types = dataframe.dtypes.to_dict()

        # Impute missing values.
        # TODO(Rui): Create maybe a dict for this.
        if config.missing_imputation == configs.MISSING_VALUES_IMPUTATION.MEAN:
            for column in dataframe.columns:
                if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].mode()[0]
                    )
                else:
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].mean()
                    )
        elif config.missing_imputation == configs.MISSING_VALUES_IMPUTATION.MEDIAN:
            for column in dataframe.columns:
                if dataframe[column].dtype not in TYPES_TO_NORMALISE:
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].mode()[0]
                    )
                else:
                    dataframe[column] = dataframe[column].fillna(
                        dataframe[column].median()
                    )
        elif config.missing_imputation == configs.MISSING_VALUES_IMPUTATION.MODE:
            for column in dataframe.columns:
                dataframe[column] = dataframe[column].fillna(
                    dataframe[column].mode()[0]
                )
        else:
            raise NotImplementedError

        # Deduplicate data.
        if config.deduplication == configs.DEDUPLICATION.KEEP:
            dataframe = dataframe.drop_duplicates(keep="first")
        else:
            raise NotImplementedError

        # Normalises dataframe.
        normalisation_holder.normalise(dataframe, config)

        # Label encode categorical data.
        if has_datetime_columns(dataframe):
            config.categorical_encoding = configs.CATEGORICAL_ENCODING.NONE

        categorical_encoder.label_encode(dataframe, config)
        return dataframe, normalisation_holder, categorical_encoder, infered_types

    @staticmethod
    def associate_dtypes(dataframe: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
        """
        Reassigns the inferred data types to the columns in the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe with columns to be cast.
        dtypes : dict
            A dictionary mapping column names to their inferred data types.

        Returns
        -------
        pd.DataFrame
            Dataframe with updated data types for each column.
        """
        dataframe_columns = dataframe.columns
        for column, dtype in dtypes.items():
            if column not in dataframe_columns:
                logger.warning("Column not found in the dataframe.", column=column)
                continue
            dataframe[column].astype(dtype)

        return dataframe

    def synthesise_data(
        self,
        dataframe: pd.DataFrame,
        features: List[str],
        sensitive_columns: Dict[str, str],
        preprocessing_config: configs.DataProcessingConfig = configs.DataProcessingConfig(),
        synthesiser_config: configs.SynthesiserConfig = configs.SynthesiserConfig(),
    ):
        """
        Synthesizes data by preprocessing, generating synthetic data,
        and anonymizing sensitive columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe to be synthesized.
        features : List[str]
            The list of feature column names for the synthesized data.
        sensitive_columns : Dict[str, str]
            Mapping of sensitive column names to their anonymized representations.
        preprocessing_config : configs.DataProcessingConfig, optional
            Configuration for data preprocessing, by default a new DataProcessingConfig instance.
        synthesiser_config : configs.SynthesiserConfig, optional
            Configuration for the synthesizer, by default a new SynthesiserConfig instance.

        Returns
        -------
        Tuple[pd.DataFrame, float]
            - Anonymized synthetic dataframe.
            - Population Stability Index (PSI) between the original and synthetic data.
        """

        normalisation_holder = NormalisationInfoHolder()
        categorical_encoder = CategoricalEncoder()
        (
            prepared_dataframe,
            normalisation_holder,
            categorical_encoder,
            infered_types,
        ) = self.prepare_data(
            dataframe, preprocessing_config, normalisation_holder, categorical_encoder
        )
        if is_time_series(prepared_dataframe):
            generated_dataframe = self.synthesise_time_series_data(
                prepared_dataframe,
                features,
                normalisation_holder,
                categorical_encoder,
                synthesiser_config,
            )
        else:
            generated_dataframe = self.synthesise_tabular_data(
                prepared_dataframe,
                features,
                normalisation_holder,
                categorical_encoder,
                synthesiser_config,
            )

        generated_dataframe = self.associate_dtypes(generated_dataframe, infered_types)

        # TODO(Rui): I do not enjoy this pattern. Need to discuss with Andre.
        # Calculate the PopulationStabilityIndex between the original and synthesised data.
        psi = PopulationStabilityIndex()
        psi_value = psi(dataframe, generated_dataframe)

        anonymised_dataframe = self.anonymizer.anonymize(
            generated_dataframe, sensitive_columns
        )

        return anonymised_dataframe, psi_value

    def synthesise_tabular_data(
        self,
        prepared_dataframe: pd.DataFrame,
        features: List[str],
        normalisation_holder: NormalisationInfoHolder,
        categorical_encoder: CategoricalEncoder,
        synthesiser_config: configs.SynthesiserConfig,
    ):
        """
        Generates synthetic tabular data using the ACTGAN model.

        Parameters
        ----------
        prepared_dataframe : pd.DataFrame
            Preprocessed input dataframe.
        features : List[str]
            List of feature column names to include in the synthetic data.
        normalisation_holder : NormalisationInfoHolder
            Holds normalization parameters for reverting scaling.
        categorical_encoder : CategoricalEncoder
            Encodes and decodes categorical features for the synthetic data.
        synthesiser_config : configs.SynthesiserConfig
            Configuration for the ACTGAN synthesizer.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the synthetic tabular data.
        """

        ACTGANSynthesizer = actgan.ACTGANSynthesizer(
            embedding_dim=32,
            generator_dim=[32, 32],
            discriminator_dim=[32, 32],
            generator_lr=0.0001,
            generator_decay=0.00001,
            discriminator_lr=0.0001,
            discriminator_decay=0.00001,
            batch_size=32,
            discriminator_steps=1,
            binary_encoder_cutoff=1,
            binary_encoder_nan_handler=None,
            cbn_sample_size=None,
            log_frequency=False,
            verbose=False,
            epochs=synthesiser_config.epochs,
            epoch_callback=None,
            pac=1,
            cuda=False,
            conditional_vector_type=ConditionalVectorType.SINGLE_DISCRETE,
            conditional_select_mean_columns=None,
            conditional_select_column_prob=None,
            reconstruction_loss_coef=0.1,
            force_conditioning=False,
        )

        ACTGANSynthesizer.fit(prepared_dataframe)

        sample_dataframe = ACTGANSynthesizer.sample(n=10)

        # Reverse the normalisation and categorical encoding.
        sample_dataframe = normalisation_holder.inverse_normalisation(sample_dataframe)
        sample_dataframe = categorical_encoder.reverse_label_encode(sample_dataframe)

        sample_dataframe = sample_dataframe[features]

        return sample_dataframe

    def synthesise_time_series_data(
        self,
        prepared_dataframe: pd.DataFrame,
        features: List[str],
        normalisation_holder: NormalisationInfoHolder,
        categorical_encoder: CategoricalEncoder,
        synthesiser_config: configs.SynthesiserConfig,
    ):
        """
        Generates synthetic time-series data using the DGAN model.

        Parameters
        ----------
        prepared_dataframe : pd.DataFrame
            Preprocessed input dataframe with time-series features.
        features : List[str]
            List of feature column names to include in the synthetic data.
        normalisation_holder : NormalisationInfoHolder
            Holds normalization parameters for reverting scaling.
        categorical_encoder : CategoricalEncoder
            Encodes and decodes categorical features for the synthetic data.
        synthesiser_config : configs.SynthesiserConfig
            Configuration for the DGAN synthesizer.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the synthetic time-series data.
        """

        # TODO(Rui/Andre): This should be defined elsewhere.
        # Filter out Timestamp columns.
        timestamp_dtypes = [
            "datetime64",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "timedelta64",
            "timedelta64[ns]",
        ]
        timestamp_columns = prepared_dataframe.select_dtypes(
            include=timestamp_dtypes
        ).columns
        prepared_dataframe = prepared_dataframe.drop(columns=timestamp_columns)

        features = prepared_dataframe.to_numpy()
        n_time_points = 2
        n = features.shape[0] // n_time_points
        features = features[: n * n_time_points, :].reshape(
            -1, n_time_points, features.shape[1]
        )

        config = DGANConfig(
            max_sequence_len=20,
            sample_len=5,
            batch_size=10,
            epochs=synthesiser_config.epochs,
        )
        dg = DGAN(config=config)

        dg.train_numpy(
            features=features,
        )

        attributes, features = dg.generate_numpy(10)
        # Converts to numpy array.
        features = np.array(features)

        # Create a dataframe with the generated data.
        sample_dataframe = pd.DataFrame(features.reshape(-1, features.shape[2]))
        sample_dataframe.columns = prepared_dataframe.columns

        # Reverse normalisation
        sample_dataframe = normalisation_holder.inverse_normalisation(sample_dataframe)
        sample_dataframe = categorical_encoder.reverse_label_encode(sample_dataframe)

        return sample_dataframe

    def identify_sensible_columns(
        self, dataframe: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Identifies and maps sensitive columns in the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe containing potentially sensitive columns.

        Returns
        -------
        Tuple[List[str], Dict[str, str]]
            - List of sensitive column names.
            - Mapping of sensitive column names to their anonymized representations.
        """
        return self.anonymizer.identify_sensible_columns(dataframe)
