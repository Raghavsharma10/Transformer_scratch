def _convert_to_from(self, to_mag, from_mag, fromVMag=None):
        """ Converts from or to V mag using the conversion tables

        :param to_mag: uppercase magnitude letter i.e. 'V' or 'K'
        :param from_mag: uppercase magnitude letter i.e. 'V' or 'K'
        :param fromVMag: MagV if from_mag is 'V'

        :return:  estimated magnitude for to_mag from from_mag
        """
        lumtype = self.spectral_type.lumType

        # rounds decimal types, TODO perhaps we should interpolate?
        specClass = self.spectral_type.roundedSpecClass

        if not specClass:  # TODO investigate implications of this
            raise ValueError('Can not convert when no spectral class is given')

        if lumtype not in ('V', ''):
            raise ValueError("Can only convert for main sequence stars. Got {0} type".format(lumtype))

        if to_mag == 'V':
            col, sign = self.column_for_V_conversion[from_mag]

            try:  # TODO replace with pandas table
                offset = float(magDict[specClass][col])
            except KeyError:
                raise ValueError('No data available to convert those magnitudes for that spectral type')

            if math.isnan(offset):
                raise ValueError('No data available to convert those magnitudes for that spectral type')
            else:
                from_mag_val = self.__dict__['mag'+from_mag]  # safer than eval
                if isNanOrNone(from_mag_val):
                    # logger.debug('2 '+from_mag)
                    raise ValueError('You cannot convert from a magnitude you have not specified in class')
                return from_mag_val + (offset*sign)
        elif from_mag == 'V':
            if fromVMag is None:
                # trying to second guess here could mess up a K->B calulation by using the intermediate measured V. While
                # this would probably be preferable it is not was was asked and therefore could give unexpected results
                raise ValueError('Must give fromVMag, even if it is self.magV')

            col, sign = self.column_for_V_conversion[to_mag]
            try:
                offset = float(magDict[specClass][col])
            except KeyError:
                raise ValueError('No data available to convert those magnitudes for that spectral type')

            if math.isnan(offset):
                raise ValueError('No data available to convert those magnitudes for that spectral type')
            else:
                return fromVMag + (offset*sign*-1)  # -1 as we are now converting the other way
        else:
            raise ValueError('Can only convert from and to V magnitude. Use .convert() instead')