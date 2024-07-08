import os
import sys
import json
import argparse
import tensorflow as tf
from eai.data import ENDPOINTS
from eai.version import __version__
from eai.utils import Logger, create_web3_account


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        action='store',
        type=str,
        choices=[
            'version',
            'init',
            'update',
            'export-model',
        ],
        help="primary command to run eai"
    )
    parser.add_argument(
        "--private-key",
        action='store',
        type=str,
        help="private key for on-chain deployment"
    )
    parser.add_argument(
        "--endpoint-domain",
        action='store',
        default="mainnet",
        choices=ENDPOINTS.keys(),
        type=str,
        help="endpoint domain for on-chain deployment",
    )
    parser.add_argument(
        "--model-inference-cost",
        action='store',
        default=0,
        type=float,
        help="model inference cost"
    )
    parser.add_argument(
        "--chunk-len",
        action='store',
        default=30000,
        type=int,
        help="chunk length for model deployment"
    )
    parser.add_argument(
        "--model",
        action='store',
        type=str,
        help="path to the model"
    )
    parser.add_argument(
        "--vocabulary",
        action='store',
        default=None,
        type=str,
        help="path to the vocabulary file"
    )
    parser.add_argument(
        "--outputs-dir",
        action='store',
        default='outputs',
        type=str,
        help="path to the output directory"
    )
    return parser.parse_known_args()


def initialize(**kwargs):
    if not os.path.exists(".env"):
        Logger.info("Creating .env file ...")
    else:
        Logger.error(
            ".env file already exists. Please use 'eai update' command to update .env file.")
        sys.exit(2)
    if kwargs['private-key'] is None:
        Logger.info("private key not provided, creating new account ...")
        account = create_web3_account()
        kwargs['private-key'] = account["private_key"]
        Logger.success(
            f"Account created successfully.\nPrivate key: {account['private_key']}.\nAddress: {account['address']}.")

    endpoints = ENDPOINTS[kwargs['endpoint-domain']]
    env_config = {
        "PRIVATE_KEY": kwargs['private-key'],
        "NODE_ENDPOINT": endpoints["node-endpoint"],
        "REGISTER_DOMAIN": endpoints["register-domain"],
        "MODEL_INFERENCE_COST": kwargs["model-inference-cost"],
        "CHUNK_LEN":  kwargs["chunk-len"],
    }
    with open(".env", "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
    Logger.success(f".env file created successfully.")


def update(**kwargs):
    if not os.path.exists(".env"):
        Logger.error(
            ".env file not found. Please use 'eai init' command to create .env file.")
        sys.exit(2)
    else:
        if kwargs['private-key'] is None:
            Logger.error(
                "private key not provided. Please provide private key to update .env file.")
            sys.exit(2)
        Logger.info("Updating .env file ...")
        endpoints = ENDPOINTS[kwargs['endpoint-domain']]
        env_config = {
            "PRIVATE_KEY": kwargs['private-key'],
            "NODE_ENDPOINT": endpoints["node-endpoint"],
            "REGISTER_DOMAIN": endpoints["register-domain"],
            "MODEL_INFERENCE_COST": kwargs["model-inference-cost"],
            "CHUNK_LEN":  kwargs["chunk-len"],
        }
        with open(".env", "w") as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
        Logger.success(".env file updated successfully.")


def export_model(**kwargs):
    """
    init the configurations for EternalAI Builder
    """
    Logger.info("Exporting model to json format for deployment ...")
    if kwargs['model'] is None:
        Logger.warning(
            '--model must be provided for command "eai export-model"')
        sys.exit(2)
    vocab = None
    if kwargs['vocabulary'] is not None:
        with open(kwargs['vocabulary'], 'r') as fid:
            vocab = json.load(fid)
    model = tf.keras.models.load_model(kwargs['model'])
    from eai.exporter import ModelExporter
    ModelExporter().export_model(model, vocabulary=vocab,
                                 output_dir=kwargs['outputs_dir'])


@Logger.catch
def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        Logger.warning(f'unknown command or argument: {arg}')
        sys.exit(2)

    # handle different primaryc commands
    if known_args.command == "version":
        Logger.success(f"✨ EternalAI Toolkit - Version: {__version__} ✨")
    elif known_args.command == 'init':
        # initialization
        args = {
            'private-key': known_args.private_key,
            'endpoint-domain': known_args.endpoint_domain,
            'model-inference-cost': known_args.model_inference_cost,
            'chunk-len': known_args.chunk_len,
        }
        initialize(**args)
    elif known_args.command == 'update':
        # update configurations
        args = {
            'private-key': known_args.private_key,
            'endpoint-domain': known_args.endpoint_domain,
            'model-inference-cost': known_args.model_inference_cost,
            'chunk-len': known_args.chunk_len,
        }
        update(**args)
    elif known_args.command == "export-model":
        # export model to json
        args = {
            'model': known_args.model,
            'vocabulary': known_args.vocabulary,
            'outputs_dir': known_args.outputs_dir
        }
        export_model(**args)


if (__name__ == "__main__"):
    main()
