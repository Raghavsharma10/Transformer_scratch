def get_all_regions_with_tiles(self):
        """
        Generator which yields a set of (rx, ry) tuples which describe
        all regions for which the world has tile data
        """
        for key in self.get_all_keys():
            (layer, rx, ry) = struct.unpack('>BHH', key)
            if layer == 1:
                yield (rx, ry)