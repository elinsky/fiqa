# -*- coding: utf-8 -*-
"""Config class"""

from __future__ import annotations
import json

class Config:
    """Config class which contains data, train and model hyperparameters"""

    def __init__(self, data: HelperObject, train: HelperObject, model: HelperObject):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg: dict) -> Config:
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_: dict):
        self.__dict__.update(dict_)