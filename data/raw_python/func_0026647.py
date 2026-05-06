def getTileUrlsByLatLngExtent(self, xmin, ymin, xmax, ymax, level):
        '''
        Returns a list of tile urls by extent
        '''
        # Upper-Left Tile
        tileXMin, tileYMin = self.tileUtils.convertLngLatToTileXY(xmin, ymax,
                                                                  level)

        # Lower-Right Tile
        tileXMax, tileYMax = self.tileUtils.convertLngLatToTileXY(xmax, ymin,
                                                                  level)

        tileUrls = []

        for y in range(tileYMax, tileYMin - 1, -1):
            for x in range(tileXMin, tileXMax + 1, 1):
                tileUrls.append(self.createTileUrl(x, y, level))

        return tileUrls