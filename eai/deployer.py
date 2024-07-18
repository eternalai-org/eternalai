import os
import pickle
from web3 import Web3
from web3 import Account
from eth_abi import encode
from typing import List, Tuple
from web3.contract import Contract
from eai.utils import Logger as logger
from eai.layer_config import LayerType, InputType
from web3.middleware import construct_sign_and_send_raw_middleware
import importlib
from eai.network_config import GAS_LIMIT, CHUNK_LEN, NETWORK
from eai.utils import fromFloat, index_last, getLayerType, getActivationType, getPaddingType, get_script_path
from eai.artifacts.models.FunctionalModel import CONTRACT_ARTIFACT


class LayerData:
    def __init__(self, layerType: LayerType, layerName: str, configData: str, inputIndices: List[int]):
        self.layerType = layerType
        self.layerName = layerName
        self.configData = configData
        self.inputIndices = inputIndices


class LayerConfig:
    def __init__(self, layerTypeIndex: int, layerAddress: str, inputIndices: List[int]):
        self.layerTypeIndex = layerTypeIndex
        self.layerAddress = layerAddress
        self.inputIndices = inputIndices
        
    def toContractParams(self):
        return (self.layerTypeIndex, self.layerAddress, self.inputIndices)


class ModelDeployer():
    def __init__(self, network: str = None):
        network_mode = network if network is not None else os.environ["NETWORK_MODE"]
        node_endpoint = NETWORK[network_mode]["NODE_ENDPOINT"]
        self.w3 = Web3(Web3.HTTPProvider(node_endpoint))
        self.private_key = os.environ["PRIVATE_KEY"]
        self.chunk_len = CHUNK_LEN
        try:
            self.address = Account.from_key(self.private_key).address
            self.w3.middleware_onion.add(
                construct_sign_and_send_raw_middleware(self.private_key))
        except Exception as e:
            raise Exception(f"Failed to initialize deployer: {e}")
        self.cache_data = {}
        self.cache_file = os.path.join(get_script_path(), ".cache")
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache_data = pickle.load(f)
        self.network_mode = network_mode

    def _update_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_data, f)

    def deploy_from_artifact(self) -> type[Contract]:
        tx_hash = self.w3.eth.contract(abi=CONTRACT_ARTIFACT['abi'], bytecode=CONTRACT_ARTIFACT['bytecode']).constructor().transact({
            "from": self.address,
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = receipt['contractAddress']
        logger.success(
            f'Contract has been deployed to address {contract_address}, tx={tx_hash.hex()}.')
        return contract_address

    def get_model_config(self, layers) -> Tuple[List[LayerData], int]:
        layersData = []
        totalWeights = 0
        for i in range(len(layers)):
            layer = layers[i]

            configData = ""
            layerType = getLayerType(layer['class_name'])
            logger.info(f"Layer {i}: {layer['class_name']}, type: {layerType}")
            inputIndices = list(
                map(lambda node: node['args'][0]['idx'], layer['inbound_nodes']))
            if layerType == LayerType.Dense:
                inputNode = layer['inbound_nodes'][0]['args'][0]
                inputUnits = inputNode['shape'][1]
                units = layer['layer_config']['units']
                activationFn = getActivationType(
                    layer['layer_config']['activation'])
                configData = encode(["uint8", "uint256", "uint256"], [
                                    activationFn.value, units, inputUnits])
                totalWeights += inputUnits * units + units
            elif layerType == LayerType.Flatten:
                configData = encode([], [])
            elif layerType == LayerType.Rescaling:
                n1 = fromFloat(layer['layer_config']['scale'])
                n2 = fromFloat(layer['layer_config']['offset'])
                configData = encode(["int64", "int64"], [n1, n2])
            elif layerType == LayerType.Softmax:
                configData = encode([], [])
            elif layerType == LayerType.ReLU:
                max_value = fromFloat(layer['layer_config']['max_value'])
                negative_slope = fromFloat(
                    layer['layer_config']['negative_slope'])
                threshold = fromFloat(layer['layer_config']['threshold'])
                configData = encode(["int64", "int64", "int64"], [
                                    max_value, negative_slope, threshold])
            elif layerType == LayerType.Sigmoid:
                configData = encode([], [])
            elif layerType == LayerType.Linear:
                configData = encode([], [])
            elif layerType == LayerType.Add:
                configData = encode([], [])
            elif layerType == LayerType.InputLayer:
                dim = layer['layer_config']['batch_input_shape']
                pos = index_last(dim, None)
                dim = dim[pos + 1:]
                if len(dim) == 0:
                    configData = encode(["uint8"], [InputType.Scalar.value])
                elif len(dim) == 1:
                    n = dim[0]
                    configData = encode(["uint8", "uint[]"], [
                                        InputType.Tensor1D.value, [n]])
                elif len(dim) == 2:
                    n = dim[0]
                    m = dim[1]
                    configData = encode(["uint8", "uint[]"], [
                                        InputType.Tensor2D.value, [n, m]])
                elif len(dim) == 3:
                    n = dim[0]
                    m = dim[1]
                    p = dim[2]
                    configData = encode(["uint8", "uint[]"], [
                                        InputType.Tensor3D.value, [n, m, p]])
            elif layerType == LayerType.MaxPooling2D:
                f_w = layer['layer_config']['pool_size'][0]
                f_h = layer['layer_config']['pool_size'][1]
                s_w = layer['layer_config']['strides'][0]
                s_h = layer['layer_config']['strides'][1]
                padding = layer['layer_config']['padding']

                configData = encode(["uint[2]", "uint[2]", "uint8"], [
                    [f_w, f_h],
                    [s_w, s_h],
                    getPaddingType(padding).value,
                ])
            elif layerType == LayerType.AveragePooling2D:
                f_w = layer['layer_config']['pool_size'][0]
                f_h = layer['layer_config']['pool_size'][1]
                s_w = layer['layer_config']['strides'][0]
                s_h = layer['layer_config']['strides'][1]
                padding = layer['layer_config']['padding']

                configData = encode(["uint[2]", "uint[2]", "uint8"], [
                    [f_w, f_h],
                    [s_w, s_h],
                    getPaddingType(padding).value,
                ])
            elif layerType == LayerType.Conv2D:
                inputNode = layer['inbound_nodes'][0]['args'][0]
                inputFilters = inputNode['shape'][3]
                outputFilters = layer['layer_config']['filters']
                f_w = layer['layer_config']['kernel_size'][0]
                f_h = layer['layer_config']['kernel_size'][1]
                s_w = layer['layer_config']['strides'][0]
                s_h = layer['layer_config']['strides'][1]
                padding = layer['layer_config']['padding']
                activationFn = getActivationType(
                    layer['layer_config']['activation'])

                configData = encode(["uint8", "uint", "uint", "uint[2]", "uint[2]", "uint8"], [
                    activationFn.value,
                    inputFilters,
                    outputFilters,
                    [f_w, f_h],
                    [s_w, s_h],
                    getPaddingType(padding).value,
                ])
                totalWeights += f_w * f_h * inputFilters * outputFilters + outputFilters
            elif layerType == LayerType.Embedding:
                inputDim = layer['layer_config']['input_dim']
                outputDim = layer['layer_config']['output_dim']
                configData = encode(["uint256", "uint256"], [
                                    inputDim, outputDim])
                totalWeights += inputDim * outputDim
            elif layerType == LayerType.SimpleRNN:
                inputNode = layer['inbound_nodes'][0]['args'][0]
                inputUnits = inputNode['shape'][2]
                units = layer['layer_config']['units']
                activationFn = getActivationType(
                    layer['layer_config']['activation'])
                configData = encode(["uint8", "uint256"], [
                                    activationFn.value, units])
                totalWeights += inputUnits * units + units * units + units
            elif layerType == LayerType.LSTM:
                inputNode = layer['inbound_nodes'][0]['args'][0]
                inputUnits = inputNode['shape'][2]
                units = layer['layer_config']['units']
                activationFn = getActivationType(
                    layer['layer_config']['activation'])
                recActivationFn = getActivationType(
                    layer['layer_config']['recurrent_activation'])
                configData = encode(["uint8", "uint8", "uint256", "uint256"], [
                                    activationFn.value, recActivationFn.value, units, inputUnits])
                totalWeights += inputUnits * units * 4 + units * units * 4 + units * 4

            layersData.append(LayerData(
                layerType,
                layer['class_name'],
                configData,
                inputIndices,
            ))

        return layersData, totalWeights

    def uploadModelWeights(self, model: type[Contract], weights: List[float], start_idx: int = 0):
        logger.info(
            f'Weights size: {len(weights)}, total length: {len(weights) * 32} bytes')
        txIdx = 0
        for l in range(0, start_idx, self.chunk_len):
            logger.success(
                f'Appending weights #{txIdx} has already been done...')
            txIdx += 1
        logger.info(f"Uploading weights from index {start_idx} ...")
        for l in range(start_idx, len(weights), self.chunk_len):
            weightsToUpload = list(
                map(fromFloat, weights[l: l + self.chunk_len]))
            logger.info(f'Appending weights #{txIdx}...')
            appendWeightTxHash = model.functions.appendWeights(weightsToUpload, l + 1).transact({
                "from": self.address,
                "gas": GAS_LIMIT
            })
            receipt = self.w3.eth.wait_for_transaction_receipt(
                appendWeightTxHash)
            if receipt['status'] != 1:
                raise Exception('tx failed', receipt)
            logger.success(
                f'tx: {appendWeightTxHash.hex()}, gas used: {receipt.gasUsed}.')
            txIdx += 1
        logger.success("Weights uploaded successfully.")

    def deploy_layer(self, layer_data: LayerData) -> LayerConfig:
        artifact_name = layer_data.layerName

        if not artifact_name.endswith("Layer"):
            artifact_name += "Layer"
        submodule = importlib.import_module(
            f"eai.artifacts.layers.{artifact_name}")
        artifact = getattr(submodule, "CONTRACT_ARTIFACT")
        tx_hash = self.w3.eth.contract(abi=artifact['abi'], bytecode=artifact['bytecode']).constructor(layer_data.configData).transact({
            "from": self.address,
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt['status'] != 1:
            raise Exception('tx failed', receipt)
        contract_address = receipt['contractAddress']
        tx = tx_hash.hex()
        logger.success(
            f'Layer {layer_data.layerName} has been deployed to address {contract_address}, tx={tx}.')
        return  {"layerTypeIndex": layer_data.layerType.value, "address": contract_address, "inputIndices": layer_data.inputIndices, "tx": tx}

    def deploy_model(self, model_data):
        assert self.private_key is not None, "Private key is required to deploy contract, please run command 'eai set-private-key' to set private key"
        logger.info("Deploying model to EternalAI chain ...")
        layers = model_data["model_graph"]["layers"]
        weights = model_data["weights"]
        hashed_model = model_data["hashed_model"]
        deployed_model_contract = self.cache_data.get(hashed_model, None)
        if deployed_model_contract is not None:
            logger.warning(
                "Model has already been deployed at address: " + deployed_model_contract["address"]
            )
        else:
            logger.info("Deploying FunctionalModel contract...")
            contract_address = self.deploy_from_artifact()
            deployed_model_contract = {}
            deployed_model_contract["address"] = contract_address
            hash_value = hashed_model + self.network_mode
            self.cache_data[hash_value] = deployed_model_contract
            self._update_cache()
        contract = self.w3.eth.contract(address=deployed_model_contract["address"], abi=CONTRACT_ARTIFACT['abi'])
        layersData, totalWeights = self.get_model_config(layers)
        logger.info("Deploying layer contracts...")
        layerConfigs = []
        deployed_layer_configs = deployed_model_contract.get("layer_configs", [])
        for idx, deployed_layer_config in enumerate(deployed_layer_configs):
            layerConfigs.append(
                LayerConfig(
                    deployed_layer_config["layerTypeIndex"],
                    deployed_layer_config["address"],
                    deployed_layer_config["inputIndices"]
                )
            )
            logger.success(
                f'Layer {layersData[idx].layerName} has already been deployed to address {deployed_layer_config["address"]}, tx={deployed_layer_config["tx"]}.')
        for idx in range(len(layerConfigs), len(layersData)):
            config= self.deploy_layer(layersData[idx])
            layerConfigs.append(
                LayerConfig(
                    config["layerTypeIndex"],
                    config["address"],
                    config["inputIndices"]
                )
            )
            deployed_layer_configs.append(config)
            deployed_model_contract["layer_configs"] = deployed_layer_configs
            self.cache_data[hashed_model] = deployed_model_contract
            self._update_cache()
        layerConfigParams = list(
            map(lambda x: x.toContractParams(), layerConfigs))
        appended_weights = 0
        if not deployed_model_contract.get("is_constructed", False):
            logger.info("Constructing model...")
            constructModelTxHash = contract.functions.constructModel(layerConfigParams).transact({
                "from": self.address,
                "gas": GAS_LIMIT
            })
            receipt = self.w3.eth.wait_for_transaction_receipt(
                constructModelTxHash)
            if receipt['status'] != 1:
                raise Exception('tx failed', receipt)
            deployed_model_contract["is_constructed"] = True
            self.cache_data[hashed_model] = deployed_model_contract
            self._update_cache()
            logger.success(f"Model constructed successfully, tx:  {constructModelTxHash.hex()}, gas used: {receipt.gasUsed}.")
        else:
            data = contract.functions.model().call()
            required_weights = data[1]
            assert len(weights) == required_weights, f"Expected {required_weights} weights, got {len(weights)}"
            appended_weights = data[2]
        self.uploadModelWeights(contract, weights, appended_weights)
        logger.success("Model deployed at address: " + contract.address)
        self.cache_data.pop(hashed_model)
        self._update_cache()
        return contract