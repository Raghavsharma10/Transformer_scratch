def radiance(self, band):
        """Calculate top of atmosphere radiance of Landsat 8
        as outlined here: http://landsat.usgs.gov/Landsat8_Using_Product.php
    
        L = ML * Q + AL
    
        where:
            L  = TOA spectral radiance (Watts / (m2 * srad * mm))
            ML = Band-specific multiplicative rescaling factor from the metadata
                 (RADIANCE_MULT_BAND_x, where x is the band number)
            AL = Band-specific additive rescaling factor from the metadata
                 (RADIANCE_ADD_BAND_x, where x is the band number)
            Q  = Quantized and calibrated standard product pixel values (DN)
                 (ndarray img)
    
        Returns
        --------
        ndarray:
            float32 ndarray with shape == input shape
    """
        ml = getattr(self, 'radiance_mult_band_{}'.format(band))
        al = getattr(self, 'radiance_add_band_{}'.format(band))
        dn = self._get_band('b{}'.format(band))
        rad = ml * dn.astype(float32) + al

        return rad