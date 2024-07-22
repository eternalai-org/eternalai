import json
import hashlib
import importlib
from eai.utils import Logger


class ModelExporter:
    def __init__(self):
        pass

    def _export_inbound_nodes_keras2(self, layer, layer_indices):
        """
        Export inbound nodes of a given layer.

        Args:
            layer (dict): The layer configuration.
            layer_indices (list): List of layer names.

        Returns:
            list: List of inbound node data for Keras 2.
        """
        ret = []
        build_config = layer.get("build_config", {"input_shape": []})
        inbound_nodes = layer["inbound_nodes"]
        inputs = build_config["input_shape"]
        if len(inputs) > 1:
            if isinstance(inputs[0], list):
                for idx, input_shape in enumerate(inputs):
                    node = inbound_nodes[0]
                    inbound_node_data = {"args": [], "kwargs": {}}
                    args = node[idx]
                    inbound_node_data["args"].append({
                        "name": args[0],
                        "idx": layer_indices.index(args[0]),
                        "shape": input_shape
                    })
                    ret.append(inbound_node_data)
            else:
                node = inbound_nodes[0]
                inbound_node_data = {"kwargs": {}, "args": []}
                args = node[0]
                inbound_node_data["args"].append({
                    "name": args[0],
                    "idx": layer_indices.index(args[0]),
                    "shape": inputs
                })
                ret.append(inbound_node_data)
        return ret

    def _export_inbound_nodes_keras3(self, layer, layer_indices):
        """
        Export inbound nodes of a given layer for Keras 3.

        Args:
            layer (dict): The layer configuration.
            layer_indices (list): List of layer names.

        Returns:
            list: List of inbound node data.
        """
        ret = []
        build_config = layer.get("build_config", {"input_shape": []})
        inbound_nodes = layer["inbound_nodes"]
        inputs = build_config["input_shape"]
        if len(inputs) > 1:
            if isinstance(inputs[0], list):
                for idx, input_shape in enumerate(inputs):
                    node = inbound_nodes[0]
                    inbound_node_data = {"args": [], "kwargs": node["kwargs"]}
                    args = node['args'][0][idx]
                    inbound_node_data["args"].append({
                        "name": args['config']["keras_history"][0],
                        "idx": layer_indices.index(args['config']["keras_history"][0]),
                        "shape": input_shape
                    })
                    ret.append(inbound_node_data)
            else:
                node = inbound_nodes[0]
                inbound_node_data = {"args": [], "kwargs": node["kwargs"]}
                for args in node['args']:
                    inbound_node_data["args"].append({
                        "name": args['config']["keras_history"][0],
                        "idx": layer_indices.index(args['config']["keras_history"][0]),
                        "shape": inputs
                    })
                ret.append(inbound_node_data)

        return ret

    def _export_model_graph(self, model):
        """
        Export the model graph.

        Args:
            model (object): The model object.
            vocabulary (str): The vocabulary.
            output_path (str): The output path.

        Returns:
            dict: The exported model graph.
        """
        # Convert the model to JSON format
        model_data = json.loads(model.to_json())
        Logger.info("Exporting model graph ...")
        graph = {
            "layers": []
        }
        layer_indices = []
        index_layer = 0
        for idx, layer in enumerate(model_data["config"]["layers"]):
            layer_config = layer["config"]
            layer_name = layer_config["name"]
            class_name = layer["class_name"]
            data = {
                "idx": idx,
                "name": layer_name,
                "class_name": class_name,
            }
            try:
                # Get the layer class and its configuration
                module = importlib.import_module("eai.layers")
                layer_class = getattr(module, class_name)(layer_config)
                layer_config = layer_class.get_layer_config()
                Logger.success(f"Layer {class_name} exported")
            except:
                Logger.error(f"Layer {class_name} not supported")
                raise Exception(f"Layer {class_name} not supported")
            data["layer_config"] = layer_config
            layer_indices.append(layer_name)
            if model_data["class_name"] == "Functional":
                keras_version = importlib.import_module("keras").__version__
                if keras_version.startswith("2."):
                    data["inbound_nodes"] = self._export_inbound_nodes_keras2(
                        layer, layer_indices)
                elif keras_version.startswith("3."):
                    data["inbound_nodes"] = self._export_inbound_nodes_keras3(
                        layer, layer_indices)
                else:
                    raise Exception("Keras version not supported")
            elif model_data["class_name"] == "Sequential":
                if idx == 0:
                    data["inbound_nodes"] = []
                    if data["class_name"] == "InputLayer":
                        prev_layer_out_shape = layer_config["batch_input_shape"]
                        index_layer = 0
                    else:
                        prev_layer_out_shape = model.layers[index_layer].output.shape
                else:
                    data["inbound_nodes"] = [{
                        "args": [
                            {
                                "name": layer_indices[idx - 1],
                                "idx": idx - 1,
                                "shape": prev_layer_out_shape
                            }
                        ],
                        "kwargs": {}
                    }]
                    prev_layer_out_shape = model.layers[index_layer].output.shape
                    index_layer += 1
            graph["layers"].append(data)
        Logger.success("Model graph exported.")
        return graph

    def _export_weights(self, model):
        """
        Export the model weights.

        Args:
            model (object): The model object.
            output_path (str): The output path.

        Returns:
            list: List of flattened weights.
        """
        Logger.info("Exporting model weights ...")
        weights = []
        flattened_weights = []
        for layer in model.layers:
            w = layer.get_weights()
            weights.append(w)
        for idx, layer in enumerate(weights):
            for weight_group in layer:
                flatten = weight_group.reshape(-1).tolist()
                for i in flatten:
                    flattened_weights.append(float(i))
        Logger.success("Weights exported.")
        return flattened_weights

    def _export_tf_model(self, model):
        """
        Export a Tensorflow/Keras model.

        Args:
            model (tf.keras.Model): The model to be exported.
            vocabulary (list, optional): The vocabulary used by the model. Defaults to None.
            output_dir (str, optional): The directory to save the exported model. Defaults to None.

        Returns:
            dict: A dictionary containing the exported model graph and weights.
        """
        Logger.info("Exporting Tensorflow/Keras model ...")

        # Export the model graph
        model_graph = self._export_model_graph(model)
        # Export the weights
        weights = self._export_weights(model)
        # Create a hash of the model graph
        hashed_graph = hashlib.sha256(json.dumps(
            model_graph, sort_keys=True).encode()).hexdigest()

        # Create a hash of the weights
        hashed_weights = hashlib.sha256(
            json.dumps(weights).encode()).hexdigest()

        # Concatenate the two hashes
        hashed_model = f"{hashed_graph}{hashed_weights}"
        Logger.success("Model exported successfully.")
        return {
            "model_graph": model_graph,
            "weights": weights,
            "hashed_model": hashed_model
        }

    def export_model(self, model):
        """
        Export a Tensorflow/Keras model.

        Args:
            model (tf.keras.Model): The model to be exported.
            vocabulary (list, optional): The vocabulary used by the model. Defaults to None.
            output_dir (str, optional): The directory to save the exported model. Defaults to None.

        Returns:
            dict: A dictionary containing the exported model graph and weights.
        """
        model_data = self._export_tf_model(model)
        return model_data
