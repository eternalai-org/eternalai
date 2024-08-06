MAX_DIM = 3


class InputLayer:
    def __init__(self, cfg):
        if "batch_shape" in cfg:
            self.batch_input_shape = cfg["batch_shape"]
        elif "batch_input_shape" in cfg:
            self.batch_input_shape = cfg["batch_input_shape"]
        else:
            raise Exception(
                "batch_shape or batch_input_shape is required for InputLayer")

    def get_layer_config(self):
        return {"batch_input_shape": self.batch_input_shape}


class Rescaling:
    def __init__(self, cfg):
        self.scale = cfg["scale"]
        self.offset = cfg["offset"]

    def get_layer_config(self):
        return {"scale": self.scale, "offset": self.offset}


class Dense:
    def __init__(self, cfg):
        self.units = cfg["units"]
        self.activation = cfg["activation"]
        self.use_bias = cfg["use_bias"]

    def get_layer_config(self):
        return {"units": self.units, "activation": self.activation, "use_bias": self.use_bias}


class Rescale:
    def __init__(self, cfg):
        self.scale = cfg["scale"]
        self.offset = cfg["offset"]

    def get_layer_config(self):
        return {"scale": self.scale, "offset": self.offset}


class Conv2D:
    def __init__(self, cfg):
        self.filters = cfg["filters"]
        self.kernel_size = cfg["kernel_size"]
        self.strides = cfg["strides"]
        self.padding = cfg["padding"]
        self.activation = cfg.get("activation", None)
        self.use_bias = cfg["use_bias"]

    def get_layer_config(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding, "activation": self.activation, "use_bias": self.use_bias}


class MaxPooling2D:
    def __init__(self, cfg):
        assert "pool_size" in cfg, "pool_size is required for MaxPooling2D"
        self.pool_size = cfg["pool_size"]
        assert "strides" in cfg, "strides is required for MaxPooling2D"
        self.strides = cfg["strides"]
        assert "padding" in cfg, "padding is required for MaxPooling2D"
        self.padding = cfg["padding"]

    def get_layer_config(self):
        return {"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding}


class AveragePooling2D:
    def __init__(self, cfg):
        assert "pool_size" in cfg, "pool_size is required for MaxPooling2D"
        self.pool_size = cfg["pool_size"]
        assert "strides" in cfg, "strides is required for MaxPooling2D"
        self.strides = cfg["strides"]
        assert "padding" in cfg, "padding is required for MaxPooling2D"
        self.padding = cfg["padding"]

    def get_layer_config(self):
        return {"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding}


class SimpleRNN:
    def __init__(self, cfg):
        assert "units" in cfg, "units is required for SimpleRNN"
        self.units = cfg["units"]
        assert "activation" in cfg, "activation is required for SimpleRNN"
        self.activation = cfg["activation"]

    def get_layer_config(self):
        return {"units": self.units, "activation": self.activation}


class Embedding:
    def __init__(self, cfg):
        assert "input_dim" in cfg, "input_dim is required for Embedding"
        self.input_dim = cfg["input_dim"]
        assert "output_dim" in cfg, "output_dim is required for Embedding"
        self.output_dim = cfg["output_dim"]

    def get_layer_config(self):
        return {"input_dim": self.input_dim, "output_dim": self.output_dim}


class LSTM:
    def __init__(self, cfg):
        assert "units" in cfg, "units is required for LSTM"
        self.units = cfg["units"]
        assert "activation" in cfg, "activation is required for LSTM"
        self.activation = cfg["activation"]
        assert "recurrent_activation" in cfg, "recurrent_activation is required for LSTM"
        self.recurrent_activation = cfg["recurrent_activation"]

    def get_layer_config(self):
        return {"units": self.units, "activation": self.activation, "recurrent_activation": self.recurrent_activation}


class Add:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class Linear:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class Sigmoid:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class ReLU:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class Softmax:
    def __init__(self, cfg):
        self.axis = -1

    def get_layer_config(self):
        return {"axis": self.axis}


class Flatten:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class Activation:
    def __init__(self, cfg):
        activation_name = cfg.get("activation", None)
        if activation_name == "relu":
            self.activation = ReLU(cfg)
        elif activation_name == "sigmoid":
            self.activation = Sigmoid(cfg)
        elif activation_name == "softmax":
            self.activation = Softmax(cfg)
        elif activation_name == "linear":
            self.activation = Linear(cfg)
        else:
            raise Exception(f"Activation {activation_name} is not supported")

    def get_layer_config(self):
        return self.activation.get_layer_config()


class BatchNormalization:
    def __init__(self, cfg):
        self.input_dim = cfg["input_dim"]
        self.momentum = cfg["momentum"]
        self.epsilon = cfg["epsilon"]

    def get_layer_config(self):
        return {"input_dim": self.input_dim, "momentum": self.momentum, "epsilon": self.epsilon}


class Dropout:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class ZeroPadding2D:
    def __init__(self, cfg):
        assert "padding" in cfg, "padding is required for ZeroPadding2D"
        self.padding = self._parse_padding(cfg["padding"])
        self.data_format = cfg["data_format"]
        assert "data_format" in cfg, "data_format is required for ZeroPadding2D"

    def _parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            if isinstance(padding[0], int):
                padding = [padding[0], padding[0], padding[1], padding[1]]
            else:
                padding = [padding[0][0], padding[0]
                           [1], padding[1][0], padding[1][1]]
        else:
            raise Exception("Invalid padding format")
        return padding

    def get_layer_config(self):
        return {"padding": self.padding, "data_format": self.data_format}


class GlobalAveragePooling2D:
    def __init__(self, cfg):
        pass

    def get_layer_config(self):
        return {}


class Concatenate:
    def __init__(self, cfg):
        assert "axis" in cfg, "axis is required for Concatenate"
        self.axis = min(cfg["axis"], MAX_DIM - 1)

    def get_layer_config(self):
        return {"axis": self.axis}
