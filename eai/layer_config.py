from enum import Enum

LayerType = Enum('LayerType', [
    'InputLayer',
    'Dense',
    'Flatten',
    'Rescaling',
    'MaxPooling2D',
    'AveragePooling2D',
    'Conv2D',
    'Embedding',
    'SimpleRNN',
    'LSTM',
    'Softmax',
    'Sigmoid',
    'ReLU',
    'Linear',
    'Add',
], start=0)

InputType = Enum('InputType', [
    'Scalar',
    'Tensor1D',
    'Tensor2D',
    'Tensor3D',
], start=0)

Activation = Enum('Activation', [
    'leakyrelu',
    'linear',
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
], start=0)

Padding = Enum('Padding', [
    'valid',
    'same',
], start=0)
