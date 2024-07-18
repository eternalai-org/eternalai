import os
import sys
import argparse
import requests 
import numpy as np
from eai.version import __version__
from eai.utils import Logger, ENV_PATH
from eai.utils import create_web3_account, publisher
from eai.network_config import NETWORK
from eai.func import transform, check, get_model

ETHER_PER_WEI = 10**18
DEFAULT_NETWORK = "testnet"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        action='store',
        type=str,
        choices=[
            'version',
            'wallet',
            'eternal',
            'call'
        ],
        help="command to execute"
    )
    # First parse the command
    args, unknown = parser.parse_known_args()
    if args.command in ['wallet', 'eternal', 'call']:
        parser.add_argument(
            "subcommand",
            action='store',
            type=str,
            help="subcommand to execute for wallet"
        )
    parser.add_argument(
        "--private-key",
        "-p",
        action='store',
        type=str,
        help="restore from private key"
    )
    parser.add_argument(
        "--format",
        action='store',
        type=str,
        default = 'keras3',
        choices=['keras2', 'keras3'],
        help="format of the model to deploy on-chain"
    )
    parser.add_argument(
        "--file",
        "-f",
        action='store',
        default = None,
        type=str,
        help="path to the model file"
    )
    parser.add_argument(
        "--url",
        action='store',
        default = None,
        type=str,
        help="url to the model file"
    )
    parser.add_argument(
        "--name",
        "-n",
        action='store',
        type=str,
        default="Unnamed Model",
        help="name of the model"
    )
    parser.add_argument(
        "--network",
        action='store',
        type=str,
        default = None,
        help="network mode"
    )
    parser.add_argument(
        "--input",
        action='store',
        type=str,
        help="input data for 'eai call predict' command"
    )
    parser.add_argument(
        "--eternal-address",
        action='store',
        type=str,
        help="address of the model for 'eai call command'"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        action='store',
        default = None,
        type=str,
        help="output path for commands"
    )

    return parser.parse_known_args()

def create_wallet(**kwargs):
    account = create_web3_account()
    Logger.success(f"Private key: {account['private_key']}")
    Logger.success(f"Address: {account['address']}")
    env_config = {
        "PRIVATE_KEY": account['private_key'],
        "NETWORK_MODE": kwargs['network'] if kwargs['network'] is not None else DEFAULT_NETWORK
    }
    with open(ENV_PATH, "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = str(value)
    Logger.success("Private key created and set successfully.")

def import_wallet(**kwargs):
    if kwargs['private-key'] is None:
        Logger.warning(
            '--private-key must be provided for command "eai wallet import"')
        sys.exit(2)
    env_config = {
        "PRIVATE_KEY": kwargs['private-key'],
        "NETWORK_MODE": kwargs['network'] if kwargs['network'] is not None else DEFAULT_NETWORK
    }
    with open(ENV_PATH, "w") as f:
        for key, value in env_config.items():
            f.write(f"{key}={value}\n")
            os.environ[key] = str(value)
    Logger.success("Private key set successfully.")  

def wallet_balance():
    if os.environ["PRIVATE_KEY"] is None:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    balance = {
        "testnet": 0,
        "mainnet": 0
    }
    # get balance on testnet
    tesnet_explorer = NETWORK["testnet"]["EXPLORER_ENDPOINT"]
    testnet_url = f"{tesnet_explorer}?module=account&action=balancemulti&address={publisher()}"
    response = requests.get(testnet_url)
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", None)
        if result is not None:
            balance["testnet"] = result[0]["balance"]
    balance["testnet"] = float(balance["testnet"]) / ETHER_PER_WEI
    # get balance on mainnet
    mainnet_explorer = NETWORK["mainnet"]["EXPLORER_ENDPOINT"]
    testnet_url = f"{mainnet_explorer}?module=account&action=balancemulti&address={publisher()}"
    response = requests.get(testnet_url)
    if response.status_code == 200:
        data = response.json()
        result = data.get("result", None)
        if result is not None:
            balance["mainnet"] = result[0]["balance"]
    balance["mainnet"] = float(balance["mainnet"]) / ETHER_PER_WEI
    Logger.success(f"Testnet: {balance['testnet']}")
    Logger.success(f"Mainnet: {balance['mainnet']}")
    return balance

def wallet_transactions(**kwargs):
    if os.environ["PRIVATE_KEY"] is None:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    network = kwargs["network"] if kwargs["network"] is not None else os.environ["NETWORK_MODE"]
    transactions = []
    explorer_endpoint = NETWORK[network]["EXPLORER_ENDPOINT"]
    url = f"{explorer_endpoint}?module=account&action=txlist&address={publisher()}&sort=desc"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        transactions = data.get("result", []) 
    for idx, transaction in enumerate(transactions):
        tx_hash = transaction["hash"]
        timeStamp = transaction["timeStamp"]
        Logger.success(f"tx: {tx_hash}, timestamp: {timeStamp}.")
    if kwargs["output-path"] is not None:
        with open(kwargs["output-path"], "w") as f:
            for idx, transaction in enumerate(transactions):
                tx_hash = transaction["hash"]
                timeStamp = transaction["timeStamp"]
                f.write(f"tx: {tx_hash}, timestamp: {timeStamp}.\n")
        Logger.success(f"Transactions written to {kwargs['output-path']}.")
            
def wallet_faucet(**kwargs):
    if os.environ["PRIVATE_KEY"] is None:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    network = kwargs["network"] if kwargs["network"] is not None else os.environ["NETWORK_MODE"]
    if network == "testnet":
        faucet_endpoint = NETWORK[network]["FAUCET_ENDPOINT"]
        url = f"{faucet_endpoint}/{publisher()}"
        response = requests.post(url)
        if response.status_code == 200:
            response = response.json()
            if response["status"] == 1:
                Logger.success("Faucet request successful.")
            else:
                Logger.error("Faucet request failed.")
        else:
            Logger.error("Faucet request failed.")
    else:
        Logger.warning("'eai wallet faucet' is only available on testnet.")
        sys.exit(2)
    

def eternal_transform(**kwargs):
    if os.environ["PRIVATE_KEY"] is None:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    if kwargs['file'] is None:
        if kwargs['url'] is None:
            Logger.warning(
                'Please provide either --file or --url for the model.')
            sys.exit(2)
        else:
            Logger.info(f"Downloading model from {kwargs['url']} ...")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file_name = temp_file.name + ".keras"
            os.system(f"wget {kwargs['url']} -O {temp_file_name}")
            kwargs['file'] = temp_file_name
    
    eternal = transform(kwargs['file'], kwargs['name'], kwargs['format'], network_mode = kwargs['network'])
    eternal.to_json(kwargs['output-path'])

def eternal_check(**kwargs):
    if kwargs['file'] is None:
        if kwargs['url'] is None:
            Logger.warning(
                'Please provide either --file or --url for the model.')
            sys.exit(2)
        else:
            Logger.info(f"Downloading model from {kwargs['url']} ...")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file_name = temp_file.name + ".keras"
            os.system(f"wget {kwargs['url']} -O {temp_file_name}")
            kwargs['file'] = temp_file_name
    check(kwargs['file'], kwargs['format'], kwargs['output-path'])

def eternal_list(**kwargs):
    if os.environ["PRIVATE_KEY"] is None:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    network = kwargs["network"] if kwargs["network"] is not None else os.environ["NETWORK_MODE"]
    list_model_endpoint = NETWORK[network]["LIST_MODEL_ENDPOINT"]
    url = f"{list_model_endpoint}?address={publisher()}&public_only=false&limit=20&offset=0"
    response = requests.get(url)
    deployed_models = []
    if response.status_code == 200:
        response = response.json()
        if response["status"] == 1:
            models = response["data"]
            for model_info in models:
                if model_info["status"] == "deployed":
                    name = model_info["model_name"]
                    address = model_info["model_address"]
                    id = model_info["model_id"]
                    price = 0
                    owner = model_info["owner"]["address"]
                    model_info_str = f"Model: {name}, Address: {address}, ID: {id}, Price: {price}, Owner: {owner}"
                    Logger.success(model_info_str)
                    deployed_models.append(model_info_str)

    if kwargs["output-path"] is not None:
        with open(kwargs["output-path"], "w") as f:
            for model_info_str in deployed_models:
                f.write(f"{model_info_str}\n")

def call_predict(**kwargs):
    model = get_model(kwargs['eternal-address'])
    input_data = np.load(kwargs['input'])
    output_arr = model.predict([input_data], kwargs['output-path'])
    return output_arr

@Logger.catch
def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        Logger.warning(f'unknown command or argument: {arg}')
        sys.exit(2)

    if known_args.command == "version":
        Logger.success(f"✨ EternalAI Toolkit - Version: {__version__} ✨")
    elif known_args.command == "wallet":
        if known_args.subcommand == "create":
            args = {
                "network": known_args.network,
            }
            create_wallet(**args)
        elif known_args.subcommand == "import":
            args = {
                "private-key": known_args.private_key,
                "network": known_args.network
            } 
            import_wallet(**args)
        elif known_args.subcommand == "transactions":
            args = {
                "output-path": known_args.output_path,
                "network": known_args.network
            }
            wallet_transactions(**args)
        elif known_args.subcommand == "balance":
            wallet_balance()
        elif known_args.subcommand == "faucet":
            args = {
                "network": known_args.network
            }
            wallet_faucet(**args)
        elif known_args.subcommand == "help":
            Logger.info("EternalAI Wallet Commands:\n")
            Logger.info("  \033[1;32meai wallet create\033[0m")
            Logger.info("    \033[33mCreate a new wallet for EternalAI\033[0m\n")
            Logger.info("  \033[1;32meai wallet import -p <private_key>\033[0m")
            Logger.info("    \033[33mImport a wallet from a private key\033[0m\n")
            Logger.info("  \033[1;32meai wallet transactions\033[0m")
            Logger.info("    \033[33mList all transactions for the wallet\033[0m\n")
            Logger.info("  \033[1;32meai wallet balance\033[0m")
            Logger.info("    \033[33mCheck the balance of the wallet\033[0m\n")
        else:
            Logger.warning(f"Subcommand {known_args.subcommand} not work for eai {known_args.command}.")
            sys.exit(2)
    elif known_args.command == "eternal":
        if known_args.subcommand == "transform":
            args = {
                "file": known_args.file,
                "url": known_args.url,
                "format": known_args.format,
                "name": known_args.name,
                "network": known_args.network,
                "output-path": known_args.output_path
            }
            eternal_transform(**args)
        elif known_args.subcommand == "check":
            args = {
                "file": known_args.file,
                "url": known_args.url,
                "format": known_args.format,
                "output-path": known_args.output_path
            }
            eternal_check(**args)
        elif known_args.subcommand == "list":
            args = {
                "output-path": known_args.output_path,
                "network": known_args.network
            }
            eternal_list(**args)
        elif known_args.subcommand == "help":
            Logger.info("EternalAI Eternal Commands:\n")
            Logger.info("  \033[92meai eternal transform --model <model_path> --format <keras2/keras3> --name <model_name> --output-path <output_path>\033[0m")
            Logger.info("    \033[93mTransform a Keras model to the EternalAI format and store it on the chain\033[0m\n")
            Logger.info("  \033[92meai eternal check --model <model_path> --format <keras2/keras3> --output-path <output_path>\033[0m")
            Logger.info("    \033[93mCheck the compatibility of a Keras model with the EternalAI format\033[0m\n")
            Logger.info("  \033[92meai eternal list\033[0m")
            Logger.info("    \033[93mList all models available from the specified address on the EternalAI chain\033[0m\n")
        else:
            Logger.warning(f"Subcommand {known_args.subcommand} not work for eai {known_args.command}.")
            sys.exit(2)
    elif known_args.command == "call":
        if known_args.subcommand == "predict":
            args = {
                "eternal-address": known_args.eternal_address,
                "input": known_args.input,
                "output-path": known_args.output_path
            }
            call_predict(**args)
        elif known_args.subcommand == "help":
            Logger.info("EternalAI Call Commands:\n")
            Logger.info("  \033[92meai call predict --model-address <address> --input <input_data> --output-path <output_path>\033[0m")
            Logger.info("    \033[93mMake a prediction using a model on the EternalAI chain\033[0m\n")

if (__name__ == "__main__"):
    main()
