# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod
from utils.config import Config

import tensorflow as tf


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg: Config):
        self.config = Config.from_json(cfg)
        tf.random.set_seed(self.config.data.seed)

    @abstractmethod
    def load_datasets(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
