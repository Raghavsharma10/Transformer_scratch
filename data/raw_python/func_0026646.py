def getTileOrigin(self, tileX, tileY, level):
        '''
        Returns the upper-left hand corner lat/lng for a tile
        '''
        pixelX, pixelY = self.convertTileXYToPixelXY(tileX, tileY)
        lng, lat = self.convertPixelXYToLngLat(pixelX, pixelY, level)
        return (lat, lng)