def _get_or_convert_magnitude(self, mag_letter):
        """ Takes input of the magnitude letter and ouputs the magnitude fetched from the catalogue or a converted value
        :return:
        """
        allowed_mags = "UBVJIHKLMN"
        catalogue_mags = 'BVIJHK'

        if mag_letter not in allowed_mags or not len(mag_letter) == 1:
            raise ValueError("Magnitude letter must be a single letter in {0}".format(allowed_mags))

        mag_str = 'mag'+mag_letter
        mag_val = self.getParam(mag_str)

        if isNanOrNone(mag_val) and ed_params.estimateMissingValues:  # then we need to estimate it!
            # old style dict comprehension for python 2.6
            mag_dict = dict(('mag'+letter, self.getParam('mag'+letter)) for letter in catalogue_mags)
            mag_class = Magnitude(self.spectralType, **mag_dict)
            try:
                mag_conversion = mag_class.convert(mag_letter)
                # logger.debug('Star Class: Conversion to {0} successful, got {1}'.format(mag_str, mag_conversion))
                self.flags.addFlag('Estimated mag{0}'.format(mag_letter))
                return mag_conversion
            except ValueError as e:  # cant convert
                logger.exception(e)
                # logger.debug('Cant convert to {0}'.format(mag_letter))
                return np.nan
        else:
            # logger.debug('returning {0}={1} from catalogue'.format(mag_str, mag_val))
            return mag_val