def get_apex(self, lat, height=None):
        """ Calculate apex height

        Parameters
        -----------
        lat : (float)
            Latitude in degrees
        height : (float or NoneType)
            Height above the surface of the earth in km or NoneType to use
            reference height (default=None)

        Returns
        ----------
        apex_height : (float)
            Height of the field line apex in km
        """
        lat = helpers.checklat(lat, name='alat')
        if height is None:
            height = self.refh

        cos_lat_squared = np.cos(np.radians(lat))**2
        apex_height = (self.RE + height) / cos_lat_squared - self.RE

        return apex_height