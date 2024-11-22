# Greatly inspired by https://github.com/gretelai/gretel-synthetics .
import numpy as np

from synthetic_energy.logger import logger
from synthetic_energy.time_series.doppelganger.config import DGANConfig
from synthetic_energy.time_series.doppelganger.doppelganger import DGAN


def hello_world():
    logger.info("Hello, world!")

    attributes = np.random.rand(10000, 3)
    features = np.random.rand(10000, 20, 2)

    config = DGANConfig(max_sequence_len=20, sample_len=5, batch_size=1000, epochs=10)

    model = DGAN(config)

    model.train_numpy(attributes=attributes, features=features)

    synthetic_attributes, synthetic_features = model.generate_numpy(10)
    logger.info(synthetic_attributes.shape)
    logger.info(synthetic_attributes)
    logger.info(synthetic_features)


if __name__ == "__main__":
    hello_world()
