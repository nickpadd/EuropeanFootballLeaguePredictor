from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """An abstraction class to use as interface for any model classes that do not inherit from keras"""

    @abstractmethod
    def build_model(self):
        """A method to build the model"""
        pass

    @abstractmethod
    def train_model(self, train_data: Any, validation_data: Any):
        """A method to fit the model
        Args:
            train_data: Train dataset
            validation_data: Validation dataset
        """
        pass

    @abstractmethod
    def evaluate(self, evaluation_data: Any):
        """A method to fit the model
        Args:
            evaluation_data: Evaluation dataset
            batch_size: Batch size
        """
        pass

    @abstractmethod
    def predict(self, inference_data: Any):
        """A method to fit the model
        Args:
            inference_data: Data to predict
        """
        pass