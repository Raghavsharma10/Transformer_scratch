def getGroundResolution(self, latitude, level):
        '''
        returns the ground resolution for based on latitude and zoom level.
        '''
        latitude = self.clipValue(latitude, self.min_lat, self.max_lat);
        mapSize = self.getMapDimensionsByZoomLevel(level)
        return math.cos(
            latitude * math.pi / 180) * 2 * math.pi * self.earth_radius / \
               mapSize