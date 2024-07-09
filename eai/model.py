import json

class EAIModel:
    def __init__(self, metadata: dict):
        assert "model_address" in metadata, "model_address is required for EAIModel object"
        self.model_address = metadata["model_address"]
        self.price = metadata.get("price", 0)
        self.name = metadata.get("name", "Unnamed Model")
        if "publisher" in metadata:
            self.publisher = metadata["publisher"]
        else:
            self.publisher = self._get_publisher()

    def _get_publisher(self):
        return None

    def set_price(self, price: float):
        self.price = price

    def get_price(self):
        return self.price

    def set_name(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    def get_publisher(self):
        return self.publisher

    def get_address(self):
        return self.model_address

    def to_json(self, output_path):
        metadata = {
            "model_address": self.model_address,
            "price": self.price,
            "name": self.name,
            "publisher": self.publisher
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f)
