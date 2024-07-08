# Fully On-Chain AI Model Deployment

This repository guides EAI users in deploying their AI models on-chain.

## Installation
**Note: This repository requires Python 3.10.**

To get started, install the EAI package using the following command:

```bash
pip install git+https://github.com/eternalai-org/eternalai.git@v1.0.0
```

To check if the installation was successful, run the following command:

```bash
eai version
```

## Usage

### Exporting Your Model

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

1. Create or Update Your Private Key:
    Run the following command in your terminal to set or create your private key:
    ```bash
    eai init --private-key YOUR_PRIVATE_KEY
    ```
    *Note: The `private_key` parameter is optional. If not provided, a new private key will be automatically generated and saved in the `.env` file.*
    To update your private key if you already have a `.env` file, run:
    ```bash
    eai update --private-key YOUR_PRIVATE_KEY
    ```
2. Check Your Model Layers and EAI Supported Layers:
    Use the following Python code to check your model’s layers and ensure they are supported by EAI:
    ```python
    import eai
    eai.check(model)
    ```

3. Save Your Model On-Chain:
    Save your trained Keras model on-chain using the following code:
    ```python
    import eai
    model_address = eai.publish(model, model_name="lenet5")
    ```
    *Note: Ensure your model is a Keras model and has been trained before saving it on-chain.*
4. Call Your Model On-Chain:
    ```python
    import eai
    output = eai.predict(model_address, input)
    ```
    *Note: Ensure your `input` is preprocessed to match the model’s expected input format.*
    


# Need help?

Join our community at [https://eternalai.org/](https://eternalai.org/)