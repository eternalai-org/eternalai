import os
import sys
import subprocess
from web3 import Account
from loguru import logger
from typing import List
from eai.layer_config import LayerType, Activation, Padding, ZeroPaddingFormat

TENSORFLOW_KERAS2 = "2.15.1"
TENSORFLOW_KERAS3 = "2.16.1"
Logger = logger
Logger.remove()
Logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="INFO",
    colorize=True
)


def create_web3_account():
    account = Account.create()
    return {"address": account.address, "private_key": account._private_key.hex()}


def publisher():
    private_key = os.environ.get("PRIVATE_KEY", None)
    if private_key is None:
        return None
    return Account.from_key(private_key).address


def get_abi_type(dim_count: int) -> str:
    if dim_count == 1:
        return "int64[]"
    if dim_count == 2:
        return "int64[][]"
    if dim_count == 3:
        return "int64[][][]"
    if dim_count == 4:
        return "int64[][][][]"
    raise Exception("Number of dimension not supported")


def getLayerType(name: str) -> int:
    try:
        return LayerType[name]
    except:
        raise Exception("Layer type not found")


def getActivationType(name: str) -> int:
    try:
        return Activation[name]
    except:
        raise Exception("Activation function type not found")


def getPaddingType(name: str) -> int:
    try:
        return Padding[name]
    except:
        raise Exception("Padding type not found")


def getZeroPadding2DType(name: str) -> int:
    try:
        return ZeroPaddingFormat[name]
    except:
        raise Exception("ZeroPadding2D type not found")


def getConvSize(dim: List[int], size: List[int], stride: List[int], padding: str):
    out = []
    pad = []
    for i in range(dim):
        if padding == "same":
            out.append((dim[i] + stride[i] - 1) // stride[i])
            total_pad = max(size[i] - stride[i], 0) if (dim[i] %
                                                        stride[i] == 0) else max(size[i] - dim[i] % stride[i], 0)
            pad.append(total_pad // 2)
        elif padding == "valid":
            out.append((dim[i] - size[i]) // stride[i] + 1)
            pad.append(0)
    return {out, pad}


def fromFloat(num: float):
    return int(num * pow(2, 32))


def index_last(arr, item):
    for r_idx, elt in enumerate(reversed(arr)):
        if elt == item:
            return len(arr) - 1 - r_idx


def get_script_path():
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.realpath(__file__))


ENV_PATH = os.path.join(get_script_path(), ".env")


def get_keras_version():
    import subprocess
    version = "3.4.1"
    try:
        result = subprocess.run(
            ['pip', 'show', "keras"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                version = line.split()[1]
    except subprocess.CalledProcessError:
        return version
    return version


def handle_keras_version(format_version):
    keras_version = get_keras_version()
    if format_version == "keras2":
        if keras_version.startswith("3."):
            Logger.warning(
                f"Your Keras version is now {keras_version} not compatible with Keras2. Downgrading to Keras2 ...")
            subprocess.run(
                ["pip", "install", "tensorflow=={}".format(TENSORFLOW_KERAS2)])
    else:
        if keras_version.startswith("2."):
            Logger.warning(
                f"Your Keras version is now {keras_version} not compatible with Keras3. Upgrading to Keras3 ...")
            subprocess.run(
                ["pip", "install", "tensorflow=={}".format(TENSORFLOW_KERAS3)])
    import keras
    Logger.success(f"Keras version is now {keras.__version__}")


ETHER_PER_WEI = 10**18
DEFAULT_RUNTIME = "cuda"
