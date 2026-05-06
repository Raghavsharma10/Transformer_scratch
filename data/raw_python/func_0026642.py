def convertLatLngToPixelXY(self, lat, lng, level):
        '''
        returns the x and y values of the pixel corresponding to a latitude
        and longitude.
        '''
        mapSize = self.getMapDimensionsByZoomLevel(level)

        lat = self.clipValue(lat, self.min_lat, self.max_lat)
        lng = self.clipValue(lng, self.min_lng, self.max_lng)

        x = (lng + 180) / 360
        sinlat = math.sin(lat * math.pi / 180)
        y = 0.5 - math.log((1 + sinlat) / (1 - sinlat)) / (4 * math.pi)

        pixelX = int(self.clipValue(x * mapSize + 0.5, 0, mapSize - 1))
        pixelY = int(self.clipValue(y * mapSize + 0.5, 0, mapSize - 1))
        return (pixelX, pixelY)