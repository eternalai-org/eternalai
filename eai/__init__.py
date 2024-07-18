import os
from dotenv import load_dotenv
from eai.utils import publisher, Logger, ENV_PATH
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not load_dotenv(ENV_PATH):
    Logger.warning(
        "private-key not found in .env file. Please use commands 'eai wallet create' or 'eai wallet import' to create a new wallet or import an existing wallet.")
from eai.version import __version__
from eai.func import transform, check, get_model