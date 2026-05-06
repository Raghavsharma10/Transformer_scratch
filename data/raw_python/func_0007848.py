def eqCoords(self, zerolat=False):
        """ Returns the Equatorial Coordinates of this object. 
        Receives a boolean parameter to consider a zero latitude. 
        
        """
        lat = 0.0 if zerolat else self.lat
        return utils.eqCoords(self.lon, lat)