def saturation_mask(self, band, value=255):
        """ Mask saturated pixels, 1 (True) is saturated.
        :param band: Image band with dn values, type: array
        :param value: Maximum (saturated) value, i.e. 255 for 8-bit data, type: int
        :return: boolean array
        """
        dn = self._get_band('b{}'.format(band))
        mask = self.mask()
        mask = where((dn == value) & (mask > 0), True, False)

        return mask