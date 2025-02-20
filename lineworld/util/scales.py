from lineworld.util import colormaps
from matplotlib.colors import ListedColormap


class Colorscale:
    def __init__(self, d, colormap_name="viridis"):
        self.d = d

        values = getattr(colormaps, f"_{colormap_name}_data")
        self.colormap = ListedColormap(values, name=colormap_name)

    def get_color(self, value):
        a = value - self.d[0]

        if a <= 0:
            return self.colormap(0)[:-1]

        a = a / (self.d[1] - self.d[0])

        if a >= 1:
            return self.colormap(1.0)[:-1]

        return self.colormap(a)[:-1]
