def lai(self):
        """
        Leaf area index (LAI), or the surface area of leaves to surface area ground.
        Trezza and Allen, 2014
        :param ndvi: normalized difference vegetation index [-]
        :return: LAI [-]
        """
        ndvi = self.ndvi()
        lai = 7.0 * (ndvi ** 3)
        lai = where(lai > 6., 6., lai)
        return lai