# Fully On-Chain AI Model Deployment

This repository guides EAI users in deploying their AI models on-chain.

## Installation
**Note: This repository requires Python 3.10.**

To get started, install the EAI package using the following command:

```bash
pip install git+https://github.com/eternalai-org/eternalai.git
```

To check if the installation was successful, run the following command:

```bash
eai version
```

***Note: This repository uses Keras3 framework by default. If you want to downgrade to Keras2, you can run the following command:***

```bash
pip install tensorflow==2.15.1
``` 

## Setting your Private Key

To set or create your private key, run the following command in your terminal:

```bash
eai set-private-key -p YOUR_PRIVATE_KEY
```

***Notes***:
- The `-p` parameter means private key, and it is **optional**. If you don't provide a private key, the system will generate one and save it in `.env` file.
- By default endpoint and backend domain will be set to mainnet. If you want to change it to testnet, you can run the following command:
```bash
eai set-private-key -p YOUR_PRIVATE_KEY -mo testnet
```

## Exporting Your Model Using the Command Line

```bash
eai publish -m PATH_TO_MODEL -name MODEL_NAME -o OUTPUT_PATH
```

***Notes***: 
- The `-m` parameter means the path to your model file, and it is **required**. If you don't provide a model path, the system will throw an error.
- The `-name` parameter means the name of your model, and it is **optional**. If you don't provide a model name, the system will use default name as ``unknowned name``.
- The `-o` parameter means the output path of your model metadata file, and it is **optional**. If you don't provide an output path, the system will output at ``output.json``.

## Exporting Your Model Using Python

0. Build or load your AI model using the Keras framework. This is an example of Lenet5 model:
    ```python
    import keras
    from keras import layers
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(28, 28, 1)),
        # C1: (None,32,32,1) -> (None,28,28,6).
        layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1),
                      activation='tanh', padding='valid'),
        # P1: (None,28,28,6) -> (None,14,14,6).
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # C2: (None,14,14,6) -> (None,10,10,16).
        layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                      activation='tanh', padding='valid'),
        # P2: (None,10,10,16) -> (None,5,5,16).
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        # Flatten: (None,5,5,16) -> (None, 400).
        layers.Flatten(),
        # FC1: (None, 400) -> (None,120).
        layers.Dense(units=120, activation='tanh'),
        # FC2: (None,120) -> (None,84).
        layers.Dense(units=84, activation='tanh'),
        # FC3: (None,84) -> (None,10).
        layers.Dense(units=10, activation='softmax'),
    ])
    model.summary()
    ```

1. Check Your Model Layers and EAI Supported Layers:
    Use the following Python code to check your model’s layers and ensure they are supported by EAI:
    ```python
    import eai
    eai.check(model)
    ```

2. Save Your Model On-Chain or Load from a Real Address
    ## Save Your Trained Keras Model On-Chain
    You can save your trained Keras model on-chain using the eai library with the following code:
    ```python
    import eai

    # Replace 'model' with your trained Keras model and provide a model name
    eai_model = eai.publish(model, model_name="lenet5")
    ``` 
    ## Load Your Model from an On-Chain Address
    To load a model from an on-chain address, use the EAIModel class from the eai.model module:
    ```python
    from eai.model import EAIModel

    # Create an instance of EAIModel
    eai_model = EAIModel()

    # Replace '0xYOUR_ADDRESS' with the actual on-chain address of your model
    eai_model.load("0xYOUR_ADDRESS")
    ```
    *Note: Make sure to replace `0xYOUR_ADDRESS` with the real address of your model on-chain.*
3. Call Your Model On-Chain:
    ```python
    output = eai_model.predict(input)
    ```
    *Note: Ensure your `input` is preprocessed to match the model’s expected input format.*
    
# Need help?

Join our community at [https://eternalai.org/](https://eternalai.org/)