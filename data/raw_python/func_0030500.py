def _geograins(self):
        """Create a map geographic area terms to the geo grain GVid values """

        from geoid.civick import GVid

        geo_grains = {}

        for sl, cls in GVid.sl_map.items():
            if '_' not in cls.level:
                geo_grains[self.stem(cls.level)] = str(cls.nullval().summarize())

        return geo_grains