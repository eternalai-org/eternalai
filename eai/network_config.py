GAS_LIMIT = 99_000_000
CHUNK_LEN = 8000
NETWORK = {
    "testnet": {
        "NODE_ENDPOINT": "https://eternal-ai3.tc.l2aas.com/rpc",
        "REGISTER_ENDPOINT": "https://api-dojo.dev2.eternalai.org/api/dojo/register-model",
        "LIST_MODEL_ENDPOINT": "https://api-dojo.dev2.eternalai.org/api/dojo/list-training-requests",
        "MODEL_INFO_BY_ADDRESS": "https://api-dojo.dev2.eternalai.org/api/dojo/model-info-by-model-address",
        "MODEL_INFO_BY_ID": "https://api-dojo.dev2.eternalai.org/api/dojo/model-info",
        "EXPLORER_ENDPOINT": "https://eternal-ai3.tc.l2aas.com/be/api",
        "FAUCET_ENDPOINT": "https://api-dojo.dev2.eternalai.org/api/service/faucet-testnet"
    },
    "mainnet": {
        "NODE_ENDPOINT": "https://node.eternalai.org",
        "REGISTER_ENDPOINT": "https://api-dojo2.eternalai.org/api/dojo/register-model",
        "LIST_MODEL_ENDPOINT": "https://api-dojo2.eternalai.org/api/dojo/list-training-requests",
        "MODEL_INFO_BY_ADDRESS": "https://api-dojo2.eternalai.org/api/dojo/model-info-by-model-address",
        "MODEL_INFO_BY_ID": "https://api-dojo2.eternalai.org/api/dojo/model-info",
        "EXPLORER_ENDPOINT": "https://explorer.eternalai.org/api"
    }
}
