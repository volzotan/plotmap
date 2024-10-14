from lineworld.layers.layer import Layer


class LayerStack():

    stack: dict[str, Layer] = {}

    def __init__(self, layers: list[Layer] = []):
        self.add(layers)

    def add(self, layers: Layer | list[Layer]) -> None:
        if type(layers) is not list:
            layer = [layers]

        for l in layers:
            self.stack[l.layer_name] = l

    def get(self, layer_name: str):
        return self.stack[layer_name]