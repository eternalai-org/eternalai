from eai.utils import publisher
from eai.utils import Logger, LayerType, handle_keras_version
import json
import os
import requests
from eai.network_config import NETWORK
from eai.model import Eternal
from eai.deployer import ModelDeployer
from eai.exporter import ModelExporter
import importlib


def register(address, name, owner):
        Logger.info("Registering model ...")
        register_endpoint = NETWORK[os.environ["NETWORK_MODE"]]["REGISTER_ENDPOINT"]
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model_address": address,
                "model_name": name,
                "owner_address": owner
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

def transform(model, model_name: str = "Unnamed Model", format = "keras3", network_mode: str = None):
    assert format in ["keras2", "keras3"], "Format must be either 'keras2' or 'keras3'"
    if network_mode is None:
        network_mode = os.environ["NETWORK_MODE"]
    handle_keras_version(format)
    import keras
    import time
    start = time.time()
    Logger.info(f"Transforming model on EternalAI's {network_mode}...")
    if isinstance(model, str):
        model = keras.models.load_model(model)
    assert isinstance(model, keras.Model), "Model must be a keras model"
    try:
        model_data = ModelExporter().export_model(model)
    except Exception as e:
        Logger.error(f"Failed to export model: {e}")
        return None
    try:
        contract = ModelDeployer(network_mode).deploy_model(model_data)
    except Exception as e:
        Logger.error(f"Failed to deploy model: {e}")
        return None
    address = contract.address
    register(address, model_name, publisher())
    try:
        eai_model = Eternal()
        eai_model.load(address)
    except Exception as e:
        Logger.error(f"Failed to load model from address: {e}")
    Logger.success(
        f"Model transformed successfully on EternalAI's {network_mode}. Time taken: {time.time() - start} seconds")
    return eai_model

def check_keras_model(model, output_path: str = None):
    try:
        model_data = json.loads(model.to_json())
    except Exception as e:
        Logger.error(f"Failed to load model data: {e}")
        return

    Logger.info("Checking model layers ...")
    supported_layers = 0
    unsupported_layers = 0
    error_layers = []

    for idx, layer in enumerate(model_data.get("config", {}).get("layers", [])):
        class_name = layer.get("class_name", "Unknown")
        layer_config = layer.get("config", {})
        try:
            module = importlib.import_module("eai.layers")
            layer_class = getattr(module, class_name)(layer_config)
            Logger.success(
                f"{idx}: Layer {class_name}")
            supported_layers += 1
        except Exception as e:
            if class_name not in error_layers:
                Logger.error(
                    f"{idx}: Layer {class_name}")
                error_layers.append(class_name)
            unsupported_layers += 1
    response = {
        "status": 1
    }
    if len(error_layers) > 0:
        response["status"] = -1
        response["error"] = []
        for error_layer in error_layers:
            response["error"].append(f"Layer {error_layer} is not supported")
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(response, f)
    else:
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(response, f)

    Logger.info(
        f"Summary: {supported_layers} layers supported, {unsupported_layers} layers not supported.")
    return response
    

def check(model, format = "keras3", output_path = None):
    assert format in ["keras2", "keras3"], "Format must be either 'keras2' or 'keras3'"
    handle_keras_version(format)
    import keras
    if isinstance(model, keras.Model):
        Logger.info(
            "Model is a keras model. Checking model layers ...")
        check_keras_model(model, output_path)
    elif isinstance(model, str):
        Logger.info(f"Loading model from {model} ...")
        try:
            model = keras.models.load_model(model)
            Logger.success("Model loaded successfully.")
        except Exception as e:
            response = {
                "status": -1,
                "error": str(e)
            }
            with open(output_path, "w") as f:
                json.dump(response, f)
            raise Exception(f"Failed to load model: {e}")
        check_keras_model(model, output_path)
    else:
        raise Exception("Model must be a keras model or a path to a keras model")
        

def layers():
    return list(LayerType.__members__.keys())

def get_model(model: str):
    eternal = Eternal()
    eternal.load(model)
    return eternal