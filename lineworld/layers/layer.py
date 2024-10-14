from sqlalchemy import engine


class Layer():

    def __init__(self, layer_name: str, db: engine.Engine):
        self.layer_name = layer_name
        self.db = db

    def input(self):
        pass

    def output(self):
        pass
