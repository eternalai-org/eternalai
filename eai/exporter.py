import os
import json
import struct
import base64
import importlib
from eai.utils import Logger


class ModelExporter:
    def __init__(self):
        pass

    def _export_inbound_nodes(self, layer, layer_indices):
        """
        Export inbound nodes of a given layer.

        Args:
            layer (dict): The layer configuration.
            layer_indices (list): List of layer names.

        Returns:
            list: List of inbound node data.
        """
        ret = []
        inbound_nodes = layer["inbound_nodes"]
        for node in inbound_nodes:
            inbound_node_data = {"args": [], "kwargs": node["kwargs"]}
            for idx, args in enumerate(node["args"]):
                if isinstance(args, dict):
                    config = args["config"]
                    inbound_node_data["args"].append({
                        "name": config["keras_history"][0],
                        "idx": layer_indices.index(config["keras_history"][0]),
                        "shape": config["shape"],
                    })
                elif isinstance(args, float):
                    inbound_node_data["args"].append(args)
                elif isinstance(args, list):
                    for arg in args:
                        config = arg["config"]
                        inbound_node_data["args"].append({
                            "name": config["keras_history"][0],
                            "idx": layer_indices.index(config["keras_history"][0]),
                            "shape": config["shape"],
                        })
                else:
                    raise Exception("Inbound node args not supported")
            ret.append(inbound_node_data)
        return ret

    def _export_model_graph(self, model, vocabulary=None, output_path=None):
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
                # Export inbound nodes for Functional model
                data["inbound_nodes"] = self._export_inbound_nodes(
                    layer, layer_indices)
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
        graph["metadata"] = {
            "vocabulary": vocabulary,
        }
        if output_path:
            # Save the model graph to a JSON file
            with open(output_path, "w") as f:
                json.dump(graph, f)
            Logger.success(f"Model graph exported to {output_path}.")
        else:
            Logger.success("Model graph exported.")
        return graph

    def _export_weights(self, model, output_path=None):
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
        weight_bytes = bytearray()
        for idx, layer in enumerate(weights):
            for weight_group in layer:
                flatten = weight_group.reshape(-1).tolist()
                for i in flatten:
                    weight_bytes.extend(struct.pack("@f", float(i)))
                    flattened_weights.append(float(i))
        weight_base64 = base64.b64encode(weight_bytes).decode()
        if output_path:
            with open(output_path, "w") as f:
                f.write(weight_base64)
            Logger.success(f"Weights exported to {output_path}.")
        else:
            Logger.success("Weights exported.")
        return flattened_weights

    def _export_tf_model(self, model, vocabulary=None, output_dir=None):
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
        model_graph_path = None
        weights_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            model_graph_path = os.path.join(output_dir, "graph.json")
            weights_path = os.path.join(output_dir, "weights.txt")
        # Export the model graph
        model_graph = self._export_model_graph(
            model, vocabulary, model_graph_path)
        # Export the weights
        weights = self._export_weights(model, weights_path)
        if output_dir is not None:
            Logger.success(f"Model exported successfully at {output_dir}.")
        else:
            Logger.success("Model exported successfully.")
        return {
            "model_graph": model_graph,
            "weights": weights
        }

    def export_model(self, model, vocabulary=None, output_dir=None):
        """
        Export a Tensorflow/Keras model.

        Args:
            model (tf.keras.Model): The model to be exported.
            vocabulary (list, optional): The vocabulary used by the model. Defaults to None.
            output_dir (str, optional): The directory to save the exported model. Defaults to None.

        Returns:
            dict: A dictionary containing the exported model graph and weights.
        """
        model_data = self._export_tf_model(model, vocabulary, output_dir)
        return model_data
