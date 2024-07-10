from eai.utils import publisher, get_abi_type
from eai.utils import Logger, LayerType
import requests
import keras
import json
import os
from eai.model import EAIModel
from eai.deployer import ModelDeployer
from eai.exporter import ModelExporter
import importlib


def register(model_addr, model_name, owner):
    Logger.info("Registering model ...")
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model_address": model_addr,
            "model_name": model_name,
            "owner_address": owner
        }

        response = requests.post(
            os.environ["REGISTER_ENDPOINT"], headers=headers, json=data)
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
