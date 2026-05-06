def land_surface_temp(self):
        """
        Mean values from Allen (2007)
        :return: 
        """
        rp = 0.91
        tau = 0.866
        rsky = 1.32
        epsilon = self.emissivity(approach='tasumi')
        radiance = self.radiance(6)
        rc = ((radiance - rp) / tau) - ((1 - epsilon) * rsky)
        lst = self.k2 / (log((epsilon * self.k1 / rc) + 1))
        return lst