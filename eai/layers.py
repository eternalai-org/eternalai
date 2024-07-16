class InputLayer:
    def __init__(self, cfg):
        if "batch_shape" in cfg:
            self.batch_input_shape = cfg["batch_shape"]
        elif "batch_input_shape" in cfg:
            self.batch_input_shape = cfg["batch_input_shape"]
        else:
            raise Exception("batch_shape or batch_input_shape is required for InputLayer")

    def get_layer_config(self):
        return {"batch_input_shape": self.batch_input_shape}


class Rescaling:
    def __init__(self, cfg):
        assert "scale" in cfg, "scale is required for Rescaling"
        self.scale = cfg["scale"]
        assert "offset" in cfg, "offset is required for Rescaling"
        self.offset = cfg["offset"]

    def get_layer_config(self):
        return {"scale": self.scale, "offset": self.offset}


class Dense:
    def __init__(self, cfg):
        assert "units" in cfg, "units is required for Dense"
        self.units = cfg["units"]
        assert "activation" in cfg, "activation is required for Dense"
        self.activation = cfg["activation"]

    def get_layer_config(self):
        return {"units": self.units, "activation": self.activation}



class Rescale:
    def __init__(self, cfg):
        assert "scale" in cfg, "scale is required for Rescale"
        self.scale = cfg["scale"]
        assert "offset" in cfg, "offset is required for Rescale"
        self.offset = cfg["offset"]

    def get_layer_config(self):
        return {"scale": self.scale, "offset": self.offset}


class Conv2D:
    def __init__(self, cfg):
        assert "filters" in cfg, "filters is required for Conv2D"
        self.filters = cfg["filters"]
        assert "kernel_size" in cfg, "kernel_size is required for Conv2D"
        self.kernel_size = cfg["kernel_size"]
        assert "strides" in cfg, "strides is required for Conv2D"
        self.strides = cfg["strides"]
        assert "padding" in cfg, "padding is required for Conv2D"
        self.padding = cfg["padding"]
        self.activation = cfg.get("activation", None)

    def get_layer_config(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides, "padding": self.padding, "activation": self.activation}


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
        self.negative_slope = cfg.get("negative_slope", 0)
        self.max_value = cfg.get("max_value", 0)
        self.threshold = cfg.get("threshold", 0)

    def get_layer_config(self):
        return {"negative_slope": self.negative_slope, "max_value": self.max_value, "threshold": self.threshold}


class Softmax:
    def __init__(self, cfg):
        self.axis = cfg.get("axis", -1)

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