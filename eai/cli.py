import os
import sys
import argparse
import tensorflow as tf
from eai.version import __version__
from eai.utils import Logger, ENV_PATH
from eai.utils import create_web3_account
from eai.func import publish, check


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        action='store',
        type=str,
        choices=[
            'version',
            'set-private-key',
            'set-node-endpoint',
            'set-backend-domain',
            'publish',
            'check'
        ],
        help="primary command to run eai"
    )
    parser.add_argument(
        "--private-key",
        "-p",
        action='store',
        type=str,
        help="private key for on-chain deployment"
    )
    parser.add_argument(
        "--node-endpoint",
        "-node",
        action='store',
        type=str,
        help="node endpoint for on-chain deployment"
    )
    parser.add_argument(
        "--backend-domain",
        "-b",
        action='store',
        type=str,
        help="domain for calling backend APIs"
    )
    parser.add_argument(
        "--model",
        "-m",
        action='store',
        type=str,
        help="path to the model"
    )
    parser.add_argument(
        "--name",
        "-name",
        action='store',
        default="Unnamed Model",
        type=str,
    )
    parser.add_argument(
        "--output-path",
        "-o",
        action='store',
        default='output.json',
        type=str,
        help="path to save metadata after publishing model"
    )
    return parser.parse_known_args()


def set_private_key(**kwargs):
    private_key = kwargs['private-key']
    if private_key is None:
        Logger.warning(
            "private-key is not provided, it will be automatically generated.")
        account = create_web3_account()
        Logger.success(f"Private key: {account['private_key']}")
        Logger.success(f"Address: {account['address']}")
        private_key = account['private_key']
    Logger.info("Setting private key ...")
    env_config = {
        "PRIVATE_KEY": private_key,
        "NODE_ENDPOINT": os.environ.get("NODE_ENDPOINT", "https://node.eternalai.org"),
        "BACKEND_DOMAIN": os.environ.get("BACKEND_DOMAIN", "https://api-dojo2.eternalai.org/api/dojo")
    }
    with open(ENV_PATH, "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = str(value)
    Logger.success("Private key set successfully.")


def set_node_endpoint(**kwargs):
    if kwargs['node-endpoint'] is None:
        Logger.error("node-endpoint is not provided.")
        sys.exit(2)
    Logger.info("Setting node endpoint ...")
    if not os.path.exists(ENV_PATH):
        Logger.error(
            "private-key is not set, please set private-key first by using command 'eai set-private-key'")
        sys.exit(2)
    env_config = {
        "PRIVATE_KEY": os.environ["PRIVATE_KEY"],
        "BACKEND_DOMAIN": os.environ["BACKEND_DOMAIN"],
        "NODE_ENDPOINT": kwargs['node-endpoint'],
    }
    with open(ENV_PATH, "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = str(value)
    Logger.success("Node endpoint set successfully.")


def set_backend_domain(**kwargs):
    if kwargs['backend-domain'] is None:
        Logger.error("backend-domain is not provided.")
        sys.exit(2)
    Logger.info("Setting backend-domain ...")
    if not os.path.exists(ENV_PATH):
        Logger.error(
            "private-key is not set, please set private-key first by using command 'eai set-private-key'")
        sys.exit(2)
    env_config = {
        "PRIVATE_KEY": os.environ["PRIVATE_KEY"],
        "NODE_ENDPOINT": os.environ["NODE_ENDPOINT"],
        "BACKEND_DOMAIN": kwargs['backend-domain'],
    }
    with open(ENV_PATH, "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = str(value)
    Logger.success("Backend domain set successfully.")


def publish_model(**kwargs):
    """
    init the configurations for EternalAI Builder
    """
    Logger.info("Exporting model to json format for deployment ...")
    if kwargs['model'] is None:
        Logger.warning(
            '--model must be provided for command "eai export-model"')
        sys.exit(2)
    try:
        model = tf.keras.models.load_model(kwargs['model'])
        model.summary()
    except Exception as e:
        Logger.error(f"Failed to load model: {e}")
        sys.exit(2)
    eai_model = publish(model, kwargs['name'])
    eai_model.to_json(kwargs['output_path'])
    Logger.success(
        f"Model published successfully, metadata saved to {kwargs['output_path']}.")

def check_model(**kwargs):
    """
    check the model for deployment
    """
    response = {}
    Logger.info("Checking model for deployment ...")
    if kwargs['model'] is None:
        Logger.warning(
            '--model must be provided for command "eai check"')
        sys.exit(2)
    check(kwargs['model'], kwargs['output_path'])   


@Logger.catch
def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        Logger.warning(f'unknown command or argument: {arg}')
        sys.exit(2)

    # handle different primaryc commands
    if known_args.command == "version":
        Logger.success(f"✨ EternalAI Toolkit - Version: {__version__} ✨")
    elif known_args.command == 'set-private-key':
        # update configurations
        args = {
            'private-key': known_args.private_key,
        }
        set_private_key(**args)
    elif known_args.command == 'set-node-endpoint':
        # update configurations
        args = {
            'node-endpoint': known_args.node_endpoint,
        }
        set_node_endpoint(**args)
    elif known_args.command == "publish":
        args = {
            'model': known_args.model,
            'name': known_args.name,
            'output_path': known_args.output_path,
        }
        publish_model(**args)
    elif known_args.command == "set-backend-domain":
        # update configurations
        args = {
            'backend-domain': known_args.backend_domain,
        }
        set_backend_domain(**args)
    elif known_args.command == "check":
        args = {
            'model': known_args.model,
            'output_path': known_args.output_path,
        }
        check_model(**args)


if (__name__ == "__main__"):
    main()
