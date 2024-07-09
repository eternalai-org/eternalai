import os
from web3 import Web3
from web3 import Account
from eth_abi import encode
from typing import List, Tuple
from web3.contract import Contract
from eai.utils import fromFloat, index_last, getLayerType, getActivationType, getPaddingType
from eai.utils import Logger as logger, LayerType, InputType
from web3.middleware import construct_sign_and_send_raw_middleware
import eai.artifacts.models.FunctionalModel
import importlib
from eai.data import GAS_LIMIT, CHUNK_LEN
from eai.utils import fromFloat, index_last, getLayerType, getActivationType, getPaddingType


class LayerData:
    def __init__(self, layerType: LayerType, layerName: str, configData: str, inputIndices: List[int]):
        self.layerType = layerType
        self.layerName = layerName
        self.configData = configData
        self.inputIndices = inputIndices


class LayerConfig:
    def __init__(self, layerType: LayerType, layerAddress: str, inputIndices: List[int]):
        self.layerType = layerType
        self.layerAddress = layerAddress
        self.inputIndices = inputIndices

    def toContractParams(self):
        return (self.layerType.value, self.layerAddress, self.inputIndices)


class ModelDeployer():
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.environ["NODE_ENDPOINT"]))
        self.private_key = os.environ["PRIVATE_KEY"]
        self.chunk_len = CHUNK_LEN
        try:
            self.address = Account.from_key(self.private_key).address
            self.w3.middleware_onion.add(
                construct_sign_and_send_raw_middleware(self.private_key))
        except Exception as e:
            logger.error(f"Invalid private key: {e}")
            raise Exception("Invalid private key")

    def deploy_from_artifact(self) -> type[Contract]:
        assert self.private_key is not None, "Private key is required to deploy contract, please run command 'eai init' to set private key"
        contract_abi = eai.artifacts.models.FunctionalModel.CONTRACT_ARTIFACT['abi']
        contract_bytecode = eai.artifacts.models.FunctionalModel.CONTRACT_ARTIFACT['bytecode']

        tx_hash = self.w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode).constructor().transact({
            "from": self.address,
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = receipt['contractAddress']
        logger.info(
            f'Contract has been deployed to address {contract_address}, tx={tx_hash.hex()}.')
        model_contract = self.w3.eth.contract(
            address=contract_address, abi=contract_abi)
        return model_contract

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

    def uploadModelWeights(self, model: type[Contract], weights: List[float]):
        assert self.private_key is not None, "Private key is required to deploy contract, please run command 'eai init' to set private key"
        logger.info("Uploading weights...")
        logger.info(
            f'Weights size: {len(weights)}, total length: {len(weights) * 32} bytes')
        txIdx = 0
        for l in range(0, len(weights), self.chunk_len):
            weightsToUpload = list(
                map(fromFloat, weights[l: l + self.chunk_len]))
            appendWeightTxHash = model.functions.appendWeights(weightsToUpload).transact({
                "from": self.address,
                "gas": GAS_LIMIT
            })
            logger.info(f'Appending weights #{txIdx}...')
            receipt = self.w3.eth.wait_for_transaction_receipt(
                appendWeightTxHash)
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
        contract_address = receipt['contractAddress']
        logger.info(
            f'Layer {layer_data.layerName} has been deployed to address {contract_address}, tx={tx_hash.hex()}.')
        return LayerConfig(
            layer_data.layerType,
            contract_address,
            layer_data.inputIndices,
        )

    def deploy_model(self, model_data):
        assert self.private_key is not None, "Private key is required to deploy contract, please run command 'eai init' to set private key"
        logger.info("Deploying model to EternalAI chain ...")
        layers = model_data["model_graph"]["layers"]
        weights = model_data["weights"]
        logger.info("Deploying FunctionalModel contract...")
        contract = self.deploy_from_artifact()
        logger.info(f'Contract address: {contract.address}')
        layersData, totalWeights = self.get_model_config(layers)

        logger.info("Deploying layer contracts...")
        layerConfigs = []
        for data in layersData:
            config = self.deploy_layer(data)
            layerConfigs.append(config)

        layerConfigParams = list(
            map(lambda x: x.toContractParams(), layerConfigs))

        logger.info("Constructing model...")
        constructModelTxHash = contract.functions.constructModel(layerConfigParams).transact({
            "from": self.address,
            "gas": GAS_LIMIT
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(
            constructModelTxHash)
        logger.success(
            f'tx: {constructModelTxHash.hex()}, gas used: {receipt.gasUsed}.')
        self.uploadModelWeights(contract, weights)
        logger.success("Model deployed at address: " + contract.address)
        return contract
