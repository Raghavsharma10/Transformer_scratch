def brightness_temp(self, band, temp_scale='K'):
        """Calculate brightness temperature of Landsat 8
    as outlined here: http://landsat.usgs.gov/Landsat8_Using_Product.php

    T = K2 / log((K1 / L)  + 1)

    and

    L = ML * Q + AL

    where:
        T  = At-satellite brightness temperature (degrees kelvin)
        L  = TOA spectral radiance (Watts / (m2 * srad * mm))
        ML = Band-specific multiplicative rescaling factor from the metadata
             (RADIANCE_MULT_BAND_x, where x is the band number)
        AL = Band-specific additive rescaling factor from the metadata
             (RADIANCE_ADD_BAND_x, where x is the band number)
        Q  = Quantized and calibrated standard product pixel values (DN)
             (ndarray img)
        K1 = Band-specific thermal conversion constant from the metadata
             (K1_CONSTANT_BAND_x, where x is the thermal band number)
        K2 = Band-specific thermal conversion constant from the metadata
             (K1_CONSTANT_BAND_x, where x is the thermal band number)

    Returns
    --------
    ndarray:
        float32 ndarray with shape == input shape
    """

        if band in self.oli_bands:
            raise ValueError('Landsat 8 brightness should be TIRS band (i.e. 10 or 11)')

        k1 = getattr(self, 'k1_constant_band_{}'.format(band))
        k2 = getattr(self, 'k2_constant_band_{}'.format(band))
        rad = self.radiance(band)
        brightness = k2 / log((k1 / rad) + 1)

        if temp_scale == 'K':
            return brightness

        elif temp_scale == 'F':
            return brightness * (9 / 5.0) - 459.67

        elif temp_scale == 'C':
            return brightness - 273.15

        else:
            raise ValueError('{} is not a valid temperature scale'.format(temp_scale))