import os
from dotenv import load_dotenv
from eai.utils import publisher, Logger, ENV_PATH
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not load_dotenv(ENV_PATH):
    Logger.warning(
        ".env file not found, please run command 'eai set-private-key' to set your private key")
from eai.version import __version__
from eai.func import publish, check, layers