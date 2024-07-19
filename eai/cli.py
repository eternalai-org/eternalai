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
    parser = argparse.ArgumentParser(
        description="Tool for managing and deploying machine learning models on-chain."
    )

    subparsers = parser.add_subparsers(dest='command', help="Commands to execute for EternalAI")
    subparsers.add_parser('version', help="Show the version of the toolkit.")
    # Wallet command
    parser_wallet = subparsers.add_parser('wallet', help="Commands to execute for wallet operations")
    wallet_commands = parser_wallet.add_subparsers(dest='subcommand', help="Subcommands to execute for wallet")

    wallet_create = wallet_commands.add_parser('create', help='Create a new wallet')
    wallet_create.add_argument(
        "--network",
        action='store',
        default=DEFAULT_NETWORK,
        type=str,
        help="Network mode. Specify the network configuration. Default is {DEFAULT_NETWORK}."
    )
    wallet_import = wallet_commands.add_parser('import', help='Import a wallet from private key')
    wallet_import.add_argument(
        "--private-key",
        "-p",
        action='store',
        type=str,
        help="Private key for the wallet"
    )
    wallet_import.add_argument(
        "--network",
        action='store',
        default=None,
        type=str,
        help="Network mode. Specify the network configuration. Default is {DEFAULT_NETWORK}."
    )
    wallet_balance = wallet_commands.add_parser('balance', help='Check the balance of the wallet')
    wallet_transactions = wallet_commands.add_parser('transactions', help='List all transactions for the wallet')
    wallet_transactions.add_argument(
        "--output-path",
        "-o",
        action='store',
        default=None,
        type=str,
        help="Output path for saving transactions list."
    )
    wallet_transactions.add_argument(
        "--network",
        action='store',
        default=None,
        type=str,
        help="Network mode. Specify the network configuration."
    )

    wallet_faucet = wallet_commands.add_parser('faucet', help='Request testnet ether from the faucet')
    wallet_faucet.add_argument(
        "--network",
        action='store',
        default=DEFAULT_NETWORK,
        type=str,
        help=f"Network mode. Specify the network configuration. Default is {DEFAULT_NETWORK}."
    )

    # Eternal command
    parser_eternal = subparsers.add_parser('eternal', help='Commands to execute for eternal operations')
    eternal_commands = parser_eternal.add_subparsers(dest='subcommand', help="Subcommands to execute for eternal")

    eternal_transform = eternal_commands.add_parser('transform', help='Transform a Keras model to the EternalAI chain')
    eternal_transform.add_argument(
        "--format",
        action='store',
        type=str,
        default='keras3',
        choices=['keras2', 'keras3'],
        help="Format of the model to deploy on-chain. Default is 'keras3'. Choices: 'keras2', 'keras3'."
    )
    eternal_transform.add_argument(
        "--file",
        "-f",
        action='store',
        default=None,
        type=str,
        help="Path to the model file."
    )
    eternal_transform.add_argument(
        "--url",
        action='store',
        default=None,
        type=str,
        help="URL to the model file."
    )
    eternal_transform.add_argument(
        "--name",
        "-n",
        action='store',
        type=str,
        default="Unnamed Model",
        help="Name of the model. Default is 'Unnamed Model'."
    )
    eternal_transform.add_argument(
        "--network",
        action='store',
        type=str,
        default=None,
        help="Network mode. Specify the network configuration."
    )
    eternal_transform.add_argument(
        "--output-path",
        "-o",
        action='store',
        default=None,
        type=str,
        help="Output path for saving transformed model metadata."
    )

    eternal_list = eternal_commands.add_parser('list', help='List all deployed models')
    eternal_list.add_argument(
        "--network",
        action='store',
        default=None,
        type=str,
        help="Network mode. Specify the network configuration."
    )
    eternal_list.add_argument(
        "--output-path",
        "-o",
        action='store',
        default=None,
        type=str,
        help="Output path for saving deployed models list."
    )
    eternal_check = eternal_commands.add_parser('check', help='Check the model format')
    eternal_check.add_argument(
        "--format",
        action='store',
        type=str,
        default='keras3',
        choices=['keras2', 'keras3'],
        help="Format of the model to deploy on-chain. Default is 'keras3'. Choices: 'keras2', 'keras3'."
    )
    eternal_check.add_argument(
        "--file",
        "-f",
        action='store',
        default=None,
        type=str,
        help="Path to the model file."
    )
    eternal_check.add_argument(
        "--url",
        action='store',
        default=None,
        type=str,
        help="URL to the model file."
    )
    eternal_check.add_argument(
        "--output-path",
        "-o",
        action='store',
        default=None,
        type=str,
        help="Output path for saving check result."
    )
    # Call command
    parser_call = subparsers.add_parser('call', help='Commands to execute for call operations')
    call_commands = parser_call.add_subparsers(dest='subcommand', help="Subcommands to execute for call")

    call_predict = call_commands.add_parser('predict', help="Call the predict function of a deployed model")
    call_predict.add_argument(
        "--input",
        action='store',
        type=str,
        help="Input data for 'eai call predict' command."
    )
    call_predict.add_argument(
        "--eternal-address",
        action='store',
        type=str,
        help="Address of the model for 'eai call' command."
    )
    call_predict.add_argument(
        "--output-path",
        "-o",
        action='store',
        default=None,
        type=str,
        help="Output path for saving prediction result."
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
    Logger.success("Wallet created and set successfully.")

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
    Logger.success("Wallet imported and set successfully.")

def wallet_balance():
    if "PRIVATE_KEY" not in os.environ:
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
    response = requests.get(testnet_url).json()
    if response["message"] == "OK":
        result = response.get("result", None)
        if result is not None:
            balance["mainnet"] = result[0]["balance"]
        balance["mainnet"] = float(balance["mainnet"]) / ETHER_PER_WEI
        Logger.success(f"Testnet: {balance['testnet']}")
        Logger.success(f"Mainnet: {balance['mainnet']}")
    else:
        Logger.error("Failed to get balance.")
    return balance

def wallet_transactions(**kwargs):
    if "PRIVATE_KEY" not in os.environ:
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
    if "PRIVATE_KEY" not in os.environ:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    network = kwargs["network"] if kwargs["network"] is not None else os.environ["NETWORK_MODE"]
    if network == "testnet":
        faucet_endpoint = NETWORK[network]["FAUCET_ENDPOINT"]
        url = f"{faucet_endpoint}/{publisher()}"
        response = requests.post(url).json()
        if response["status"] == 1:
            Logger.success("Faucet request successful.")
        else:
            error = response.get("error", "Something went wrong.")
            Logger.error(f"Faucet request failed: {error}")
    else:
        Logger.warning("'eai wallet faucet' is only available on testnet.")
        sys.exit(2)
    
def handle_wallet(args):
    if args.subcommand == "create":
        kwargs = {
            "network": args.network
        }
        create_wallet(**kwargs)
    elif args.subcommand == "import":
        kwargs = {
            "private-key": args.private_key,
            "network": args.network
        }
        import_wallet(**kwargs)
    elif args.subcommand == "transactions":
        kwargs = {
            "output-path": args.output_path,
            "network": args.network
        }
        wallet_transactions(**kwargs)
    elif args.subcommand == "balance":
        wallet_balance()
    elif args.subcommand == "faucet":
        kwargs = {
            "network": args.network
        }
        wallet_faucet(**kwargs)
    else:
        Logger.warning(f"Subcommand {args.subcommand} not work for eai {args.command}.")
        sys.exit(2)

def eternal_transform(**kwargs):
    if "PRIVATE_KEY" not in os.environ:
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
    if "PRIVATE_KEY" not in os.environ:
        Logger.warning(
            "Private key not found. Please use command 'eai wallet create' or 'eai wallet import' to create a new wallet or restore from private key.")
        sys.exit(2)
    network = kwargs["network"] if kwargs["network"] is not None else os.environ["NETWORK_MODE"]
    list_model_endpoint = NETWORK[network]["LIST_MODEL_ENDPOINT"]
    url = f"{list_model_endpoint}?address={publisher()}&public_only=false&limit=20&offset=0"
    response = requests.get(url).json()
    deployed_models = []
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
    else:
        Logger.error(f"Failed to get deployed models from EternalAI's {network}.")

def handle_eternal(args):
    if args.subcommand == "transform":
        kwargs = {
            "file": args.file,
            "url": args.url,
            "name": args.name,
            "format": args.format,
            "network": args.network,
            "output-path": args.output_path
        }
        eternal_transform(**kwargs)
    elif args.subcommand == "check":
        kwargs = {
            "file": args.file,
            "url": args.url,
            "format": args.format,
            "output-path": args.output_path
        }
        eternal_check(**kwargs)
    elif args.subcommand == "list":
        kwargs = {
            "network": args.network,
            "output-path": args.output_path
        }
        eternal_list(**kwargs)
    else:
        Logger.warning(f"Subcommand {args.subcommand} not work for eai {args.command}.")
        sys.exit(2)

def call_predict(**kwargs):
    model = get_model(kwargs['eternal-address'])
    input_data = np.load(kwargs['input'])
    output_arr = model.predict([input_data], kwargs['output-path'])
    return output_arr

def handle_call(args):
    if args.subcommand == "predict":
        kwargs = {
            "input": args.input,
            "eternal-address": args.eternal_address,
            "output-path": args.output_path
        }
        call_predict(**kwargs)
    else:
        Logger.warning(f"Subcommand {args.subcommand} not work for eai {args.command}.")
        sys.exit(2)

@Logger.catch
def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        Logger.warning(f'unknown command or argument: {arg}')
        sys.exit(2)

    if known_args.command == "version":
        Logger.success(f"✨ EternalAI Toolkit - Version: {__version__} ✨")
    elif known_args.command == "wallet":
        handle_wallet(known_args)
    elif known_args.command == "eternal":
        handle_eternal(known_args)
    elif known_args.command == "call":
        handle_call(known_args)
        
if (__name__ == "__main__"):
    main()
