def convertPixelXYToLngLat(self, pixelX, pixelY, level):
        '''
        converts a pixel x, y to a latitude and longitude.
        '''
        mapSize = self.getMapDimensionsByZoomLevel(level)
        x = (self.clipValue(pixelX, 0, mapSize - 1) / mapSize) - 0.5
        y = 0.5 - (self.clipValue(pixelY, 0, mapSize - 1) / mapSize)

        lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        lng = 360 * x

        return (lng, lat)