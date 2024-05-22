import yaml
import os

class MetadataHandler:
    def __init__(self, fn):
        self.fn = fn
        self.metadata = self.load_metadata() # dict of metadata

    def load_metadata(self):
        if not os.path.exists(self.fn):
            with open(self.fn, 'w'): pass
        with open(self.fn, 'r') as file:
            try:
                metadata = yaml.safe_load(file)
                return metadata if metadata is not None else {}
            except yaml.YAMLError as exc:
                print(exc)
                return {}
    
    def set_attribute(self, key, value):
        self.metadata[key] = value
        self.save_metadata()

    def save_metadata(self):
        with open(self.fn, 'w') as file:
            yaml.dump(self.metadata, file)
        print(self.metadata)