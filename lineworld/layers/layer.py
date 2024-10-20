from sqlalchemy import engine

class Layer():

    def __init__(self, layer_name: str, db: engine.Engine):
        self.layer_name = layer_name
        self.db = db

    def transform_to_world(self):
        pass

    def transform_to_map(self):
        pass

    def transform_to_lines(self):
        pass

    def load(self):
        pass

    def out(self):
        pass
