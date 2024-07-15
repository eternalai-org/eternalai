import os
import json
import requests
import numpy as np
from typing import List
from eth_abi import encode, decode
from web3 import Web3
from eai.utils import get_abi_type
from eai.utils import Logger
from eai.artifacts.models.FunctionalModel import CONTRACT_ARTIFACT


class TensorData:
    def __init__(self, data: bytes, dim: List[int]):
        self.data = data
        self.dim = dim

    def toContractParams(self):
        return (self.data, self.dim)

    @staticmethod
    def from_numpy(arr: np.ndarray):
        arr_32x32 = (arr * (1 << 32)).astype(int)
        dim_count = len(arr_32x32.shape)
        return TensorData(encode([get_abi_type(dim_count)], [arr_32x32.tolist()]), arr_32x32.shape)

    def to_numpy(self):
        arr_32x32 = np.asarray(
            decode([get_abi_type(len(self.dim))], self.data), dtype=np.float32)
        arr = arr_32x32 / (1 << 32)
        return arr


class EAIModel:
    def __init__(self, **kwargs):
        self.address = kwargs.get("address", None)
        self.price = kwargs.get("price", 0)
        self.name = kwargs.get("name", "unnamed model")
        self.owner = kwargs.get("owner", None)

    def set_price(self, price: float):
        self.price = price

    def get_price(self):
        return self.price

    def set_name(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    def get_publisher(self):
        return self.owner
    
    def load(self, address):
        self.address = address
        self._load_metadata_from_address()

    def _load_metadata_from_address(self):
        Logger.info(f"Loading metadata from address {self.address}")
        model_info_endpoint = os.environ["BACKEND_DOMAIN"] + "/model-info-by-model-address"
        try:
            url = model_info_endpoint + "/" + self.address
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            if response.json().get("status") == 1:
                data = response.json().get("data")
                self.name = data.get("model_name")
                self.price = 0
                self.owner = data["creator"]["address"]
                Logger.success("Metadata loaded successfully.")
            else:
                Logger.error("Failed to load metadata. Response status not 1.")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")

    def get_address(self):
        return self.address

    def to_json(self, output_path = None):
        metadata = {
            "address": self.address,
            "price": self.price,
            "name": self.name,
            "owner": self.owner
        }
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(metadata, f)
        return metadata

    def predict(self, inputs: List[np.ndarray]) -> np.ndarray:
        w3 = Web3(Web3.HTTPProvider(os.environ["NODE_ENDPOINT"]))
        contract_abi = CONTRACT_ARTIFACT['abi']
        model_contract = w3.eth.contract(
            address=self.address, abi=contract_abi)
        input_tensors = list(map(lambda x: TensorData.from_numpy(x), inputs))
        input_params = list(map(lambda x: x.toContractParams(), input_tensors))
        result = model_contract.functions.predict(input_params).call()
        output_tensor = TensorData(result[0], result[1])
        return output_tensor.to_numpy()
    
    def register(self):
        Logger.info("Registering model ...")
        register_endpoint = os.environ["BACKEND_DOMAIN"] + "/register-model"
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model_address": self.address,
                "model_name": self.name,
                "owner_address": self.owner
            }
            response = requests.post(
                register_endpoint, headers=headers, json=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            if response.json().get("status") == 1:
                Logger.success("Model registered successfully.")
            else:
                Logger.error("Failed to register model. Response status not 1.")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")
