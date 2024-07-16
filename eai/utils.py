import os
import sys
from web3 import Account
from loguru import logger
from typing import List
from eai.layer_config import LayerType, Activation, Padding

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
        logger.warning(f'Layer type not found: {name}')
        raise Exception("Layer type not found")


def getActivationType(name: str) -> int:
    try:
        return Activation[name]
    except:
        logger.warning(f'Activation function type not found: {name}')
        raise Exception("Activation function type not found")


def getPaddingType(name: str) -> int:
    try:
        return Padding[name]
    except:
        logger.warning(f'Padding type not found: {name}')
        raise Exception("Padding type not found")


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
