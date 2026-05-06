def G(self, ID, lat, lon):
        """ Creates a generic entry for an object. """
        
        # Equatorial coordinates
        eqM = utils.eqCoords(lon, lat)
        eqZ = eqM
        if lat != 0:
            eqZ = utils.eqCoords(lon, 0)
        
        return {
            'id': ID,
            'lat': lat,
            'lon': lon,
            'ra': eqM[0],
            'decl': eqM[1],
            'raZ': eqZ[0],
            'declZ': eqZ[1],
        }