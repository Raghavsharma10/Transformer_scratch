def reflectance(self, band):
        """ 
        :param band: An optical band, i.e. 1-5, 7
        :return: At satellite reflectance, [-]
        """
        if band == 6:
            raise ValueError('LT5 reflectance must be other than  band 6')

        rad = self.radiance(band)
        esun = self.ex_atm_irrad[band - 1]
        toa_reflect = (pi * rad * self.earth_sun_dist ** 2) / (esun * cos(self.solar_zenith_rad))

        return toa_reflect