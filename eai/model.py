import os
import json
import numpy as np
from typing import List
from eth_abi import encode, decode
from web3 import Web3
from eai.utils import get_abi_type
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
    def __init__(self, metadata: dict = None):
        if metadata is not None:
            assert "model_address" in metadata, "model_address is required for EAIModel object"
            self.model_address = metadata["model_address"]
            self.price = metadata.get("price", 0)
            self.name = metadata.get("name", "Unnamed Model")
            if "publisher" in metadata:
                self.publisher = metadata["publisher"]
            else:
                self.publisher = self._get_publisher()

    def _get_publisher(self):
        return None

    def set_price(self, price: float):
        self.price = price

    def get_price(self):
        return self.price

    def set_name(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    def get_publisher(self):
        return self.publisher
    
    def load(self, model_address):
        self.model_address = model_address
        self._load_metadata_from_address()

    def _load_metadata_from_address(self):
        self.price = 0
        self.name = "Unnamed Model"
        self.publisher = self._get_publisher()

    def get_address(self):
        return self.model_address

    def to_json(self, output_path):
        metadata = {
            "model_address": self.model_address,
            "price": self.price,
            "name": self.name,
            "publisher": self.publisher
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f)

    def predict(self, inputs: List[np.ndarray]) -> np.ndarray:
        w3 = Web3(Web3.HTTPProvider(os.environ["NODE_ENDPOINT"]))
        contract_abi = CONTRACT_ARTIFACT['abi']
        model_contract = w3.eth.contract(
            address=self.model_address, abi=contract_abi)
        input_tensors = list(map(lambda x: TensorData.from_numpy(x), inputs))
        input_params = list(map(lambda x: x.toContractParams(), input_tensors))
        result = model_contract.functions.predict(input_params).call()
        output_tensor = TensorData(result[0], result[1])
        return output_tensor.to_numpy()
