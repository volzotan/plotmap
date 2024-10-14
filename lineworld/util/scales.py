from lineworld.util import colormaps
from matplotlib.colors import ListedColormap

class Colorscale():

    def __init__(self, d, colormap_name="viridis"):
        self.d = d

        values = getattr(colormaps, "_{}_data".format(colormap_name))
        self.colormap = ListedColormap(values, name=colormap_name)

        # print(colormap(0.5)[:-1])

        # if type(value) in [list, tuple]:
        #     print("list")
        # else:
        #     print("no list")


    def get_color(self, value):
        a = value - self.d[0]

        if (a <= 0):
            return self.colormap(0)[:-1]

        a = a / (self.d[1] - self.d[0])

        if (a >= 1):
            return self.colormap(1.0)[:-1]

        return self.colormap(a)[:-1]