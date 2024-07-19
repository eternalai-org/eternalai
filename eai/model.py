import os
import json
import requests
import numpy as np
from typing import List
from eth_abi import encode, decode
from web3 import Web3
from eai.utils import get_abi_type
from eai.utils import Logger
from eai.network_config import NETWORK
from eai.artifacts.models.FunctionalModel import CONTRACT_ARTIFACT


class TensorData:
    def __init__(self, data: bytes, dim: List[int]):
        self.data = data
        self.dim = dim

    def toContractParams(self):
        return (self.data, self.dim)

    @staticmethod
    def from_numpy(arr):
        arr_32x32 = (arr * (1 << 32)).astype(int)
        dim_count = len(arr_32x32.shape)
        return TensorData(encode([get_abi_type(dim_count)], [arr_32x32.tolist()]), arr_32x32.shape)

    def to_numpy(self):
        arr_32x32 = np.asarray(
            decode([get_abi_type(len(self.dim))], self.data), dtype=np.float32)
        arr = arr_32x32 / (1 << 32)
        return arr


class Eternal:
    def __init__(self):
        self.model_id = None
        self.address = None
        self.price = None
        self.name = None
        self.owner = None
        self.status = None

    def __str__(self):
        return f"id: {self.model_id}, address: {self.address}, name: {self.name}, price: {self.price}, owner: {self.owner}, status: {self.status}."

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
    
    def load(self, model: str):
        if model.startswith("0x"):
            checksum_address = Web3.to_checksum_address(model)
            self._load_metadata_from_address(checksum_address)
        else:
            self._load_metadata_from_id(model)

    def _load_metadata_from_id(self, id):
        self.id = id
        Logger.info(f"Loading metadata from id {self.id}")
        endpoint = NETWORK[os.environ["NETWORK_MODE"]]["MODEL_INFO_BY_ID"]
        try:
            url = f"{endpoint}/{self.id}"
            response = requests.get(url)
            response = response.json()
            if response["status"] == 1:
                self.status = response["data"]["status"]
                self.address = response["data"]["model_address"]
                self.name = response["data"]["model_name"]
                self.price = 0
                self.owner = response["data"]["owner"]["address"]
                Logger.success("Metadata loaded successfully.")
            else:
                Logger.error("Failed to load metadata. Response status not 1.")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")

    def _load_metadata_from_address(self, address: str):
        self.address = address
        Logger.info(f"Loading metadata from address {self.address}")
        endpoint = NETWORK[os.environ["NETWORK_MODE"]]["MODEL_INFO_BY_ADDRESS"]
        try:
            url = f"{endpoint}/{self.address}"
            response = requests.get(url)
            response = response.json()
            if response["status"] == 1:
                self.status = response["data"]["status"]
                self.id = response["data"]["model_id"]
                self.name = response["data"]["model_name"]
                self.price = 0
                self.owner = response["data"]["owner"]["address"]
                Logger.success("Metadata loaded successfully.")
            else:
                Logger.error("Failed to load metadata. Response status not 1.")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")

    def get_address(self):
        return self.address
    
    def get_id(self):
        return self.id

    def to_json(self, output_path = None):
        metadata = {
            "address": self.address,
            "id": self.id,
            "status": self.status,
            "price": self.price,
            "name": self.name,
            "owner": self.owner
        }
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(metadata, f)
            Logger.success(
                f"Transformed model metadata saved to {output_path}.")
        return metadata

    def predict(self, inputs: List[np.ndarray], output_path: str = None) -> np.ndarray:
        network = os.environ["NETWORK_MODE"]
        address = self.address
        Logger.info("Making prediction on EternalAI's {} at {} ...".format(network, address))
        import time
        start = time.time()
        node_endpoint = NETWORK[network]["NODE_ENDPOINT"]
        w3 = Web3(Web3.HTTPProvider(node_endpoint))
        contract_abi = CONTRACT_ARTIFACT['abi']
        model_contract = w3.eth.contract(
            address=self.address, abi=contract_abi)
        input_tensors = list(map(lambda x: TensorData.from_numpy(x), inputs))
        input_params = list(map(lambda x: x.toContractParams(), input_tensors))
        result = model_contract.functions.predict(input_params).call()
        output_tensor = TensorData(result[0], result[1])
        output_numpy = output_tensor.to_numpy()
        Logger.success("Prediction made successfully in {} seconds. Output: {}".format(
            time.time() - start, output_numpy.tolist()))
        if output_path is not None:
            np.save(output_path, output_numpy)
            Logger.success(
                f"Prediction saved to {output_path}.")
        return output_numpy
    
