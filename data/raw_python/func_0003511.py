def reflectance(self, band):
        """Calculate top of atmosphere reflectance of Landsat 8
        as outlined here: http://landsat.usgs.gov/Landsat8_Using_Product.php
    
        R_raw = MR * Q + AR
    
        R = R_raw / cos(Z) = R_raw / sin(E)
    
        Z = 90 - E (in degrees)
    
        where:
    
            R_raw = TOA planetary reflectance, without correction for solar angle.
            R = TOA reflectance with a correction for the sun angle.
            MR = Band-specific multiplicative rescaling factor from the metadata
                (REFLECTANCE_MULT_BAND_x, where x is the band number)
            AR = Band-specific additive rescaling factor from the metadata
                (REFLECTANCE_ADD_BAND_x, where x is the band number)
            Q = Quantized and calibrated standard product pixel values (DN)
            E = Local sun elevation angle. The scene center sun elevation angle
                in degrees is provided in the metadata (SUN_ELEVATION).
            Z = Local solar zenith angle (same angle as E, but measured from the
                zenith instead of from the horizon).
    
        Returns
        --------
        ndarray:
            float32 ndarray with shape == input shape
    
        """

        if band not in self.oli_bands:
            raise ValueError('Landsat 8 reflectance should OLI band (i.e. bands 1-8)')

        elev = getattr(self, 'sun_elevation')
        dn = self._get_band('b{}'.format(band))
        mr = getattr(self, 'reflectance_mult_band_{}'.format(band))
        ar = getattr(self, 'reflectance_add_band_{}'.format(band))

        if elev < 0.0:
            raise ValueError("Sun elevation must be non-negative "
                             "(sun must be above horizon for entire scene)")

        rf = ((mr * dn.astype(float32)) + ar) / sin(deg2rad(elev))

        return rf