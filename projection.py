import math

class Converter(object):

    def __init__(self, map_up_left, map_down_right, map_size):

        self.north      = map_up_left[0]
        self.south      = map_down_right[0]
        self.west       = map_up_left[1]
        self.east       = map_down_right[1]

        self.north_px   = self._convert_lat(self.north)
        self.south_px   = self._convert_lat(self.south)
        self.west_px    = self._convert_lon(self.west)
        self.east_px    = self._convert_lon(self.east)

        # northern hemisphere, ...

        self.lat_diff = self.north - self.south
        self.lat_diff_px = self.south_px - self.north_px
        self.lon_diff = self.east - self.west
        self.lon_diff_px = self.east_px - self.west_px

        self.map_size_x = map_size
        self.map_size_y = map_size * (self.lat_diff_px / self.lon_diff_px) 

    def _convert_lat(self, lat):

        latRad  = (lat * math.pi) / 180.0
        mercN   = math.log(math.tan((math.pi / 4.0) + (latRad / 2.0)))
        y       = 0.5 - (mercN / (2.0*math.pi))

        return y

    def _convert_lon(self, lon):
        return (lon + 180.0) / 360.0

    def convert(self, lat, lon):

        print("{} -- {}".format(lat, lon))

        x = ((self._convert_lon(lon) - self.west_px) / self.lon_diff_px) * self.map_size_x
        y = ((self._convert_lat(lat) - self.north_px) / self.lat_diff_px) * self.map_size_y

        return (x, y)

    def convert_px_to_latlon(self, x, y):
        lat = self.north + (self.south - self.north) + (y / self.map_size_y)
        lon = self.east + (self.west - self.east) + (x / self.map_size_x)

        return (lat, lon)

    def get_map_size(self):
        return (self.map_size_x, self.map_size_y)

    def all_elements_inside_boundary(self, coords):
        for x, y in coords:
            if x < 0:
                return False
            if x > self.map_size_x:
                return False
            if y < 0:
                return False
            if y > self.map_size_y:
                return False

        return True

    def all_elements_outside_boundary(self, coords):
        for x, y in coords:
            if x > 0 and x < self.map_size_x:
                if y > 0 and y < self.map_size_y:
                    return False

        return True

    @staticmethod
    def get_bounding_box_in_latlon(center_point, width, height):
        map_up_left     = (center_point[0] + Converter._m_to_latlon(height/2), center_point[1] - Converter._m_to_latlon(width/2))
        map_down_right  = (center_point[0] - Converter._m_to_latlon(height/2), center_point[1] + Converter._m_to_latlon(width/2))

        return (map_up_left, map_down_right)

    @staticmethod
    def _m_to_latlon(m):
        return (m / 1.1) * 0.00001