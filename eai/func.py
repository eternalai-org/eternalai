from eai.utils import publisher, get_abi_type
from typing import List
from eth_abi import encode, decode
from eai.utils import Logger, LayerType
from web3 import Web3
import numpy as np
import requests
import keras
import json
import os
from eai.model import EAIModel
from eai.data import REGISTER_DOMAIN
from eai.deployer import ModelDeployer
from eai.exporter import ModelExporter
import importlib


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


def register(model_addr, model_name, owner):
    Logger.info("Registering model ...")
    try:
        endpoint = os.path.join(
            REGISTER_DOMAIN, "api/dojo/register-model")
        headers = {"Content-Type": "application/json"}
        data = {
            "model_address": model_addr,
            "model_name": model_name,
            "owner_address": owner
        }

        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        if response.json().get("status") == 1:
            Logger.success("Model registered successfully.")
        else:
            Logger.error("Failed to register model. Response status not 1.")

    except requests.exceptions.RequestException as e:
        Logger.error(f"Failed to send request to register model: {e}")
    except Exception as e:
        Logger.error(f"An unexpected error occurred: {e}")


def publish(model: keras.Model, model_name: str = "Unnamed Model") -> EAIModel:
    import time
    start = time.time()
    assert isinstance(model, keras.Model), "Model must be a keras model"
    try:
        model_data = ModelExporter().export_model(model)
    except Exception as e:
        Logger.error(f"Failed to export model: {e}")
        return None
    try:
        contract = ModelDeployer().deploy_model(model_data)
    except Exception as e:
        Logger.error(f"Failed to deploy model: {e}")
        return None
    address = contract.address
    register(address, model_name, publisher())
    eai_model = EAIModel(
        {"model_address": address, "name": model_name, "publisher": publisher()})
    Logger.success(
        f"Model published successfully. Time taken: {time.time() - start} seconds")
    return eai_model


def predict(model_address: str, inputs: List[np.ndarray]) -> np.ndarray:
    from eai.artifacts.models.FunctionalModel import CONTRACT_ARTIFACT
    w3 = Web3(Web3.HTTPProvider(os.environ["NODE_ENDPOINT"]))
    contract_abi = CONTRACT_ARTIFACT['abi']
    model_contract = w3.eth.contract(address=model_address, abi=contract_abi)
    input_tensors = list(map(lambda x: TensorData.from_numpy(x), inputs))
    input_params = list(map(lambda x: x.toContractParams(), input_tensors))
    result = model_contract.functions.predict(input_params).call()
    output_tensor = TensorData(result[0], result[1])
    return output_tensor.to_numpy()


def check(model: keras.Model):
    assert isinstance(model, keras.Model), "Model must be a keras model"

    try:
        model_data = json.loads(model.to_json())
    except Exception as e:
        Logger.error(f"Failed to load model data: {e}")
        return

    Logger.info("Checking model layers ...")
    supported_layers = 0
    unsupported_layers = 0

    for idx, layer in enumerate(model_data.get("config", {}).get("layers", [])):
        class_name = layer.get("class_name", "Unknown")
        layer_config = layer.get("config", {})
        try:
            module = importlib.import_module("eai.layers")
            layer_class = getattr(module, class_name)(layer_config)
            Logger.success(
                f"{idx}: Layer {class_name} with this configuration could be deployed to Eternal AI chain.")
            supported_layers += 1
        except Exception as e:
            Logger.error(
                f"{idx}: Layer {class_name} with this configuration could not be deployed to Eternal AI chain.")
            unsupported_layers += 1

    Logger.info(
        f"Summary: {supported_layers} layers supported, {unsupported_layers} layers not supported.")


def layers():
    return list(LayerType.__members__.keys())
