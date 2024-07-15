from eai.utils import publisher
from eai.utils import Logger, LayerType
import keras
import json
from eai.model import EAIModel
from eai.deployer import ModelDeployer
from eai.exporter import ModelExporter
import importlib


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
    eai_model = EAIModel(**{"address": address, 
                            "name": model_name, 
                            "owner": publisher()})
    eai_model.register()
    Logger.success(
        f"Model published successfully. Time taken: {time.time() - start} seconds")
    return eai_model

def check_keras_model(model: keras.Model, output_path: str = None):
    assert isinstance(model, keras.Model), "Model must be a keras model"

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

    if len(error_layers) > 0:
        response = {
            "status": -1,
            "error": []
        }
        for error_layer in error_layers:
            response["error"].append(f"Layer {error_layer} is not supported")
        with open(output_path, "w") as f:
            json.dump(response, f)
    else:
        response = {
            "status": 1
        }
        with open(output_path, "w") as f:
            json.dump(response, f)

    Logger.info(
        f"Summary: {supported_layers} layers supported, {unsupported_layers} layers not supported.")

def check(model, output_path = None):
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
