def convert(self, to_mag, from_mag=None):
        """ Converts magnitudes using UBVRIJHKLMNQ photometry in Taurus-Auriga (Kenyon+ 1995)
         ReadMe+ftp1995ApJS..101..117K Colors for main-sequence stars

         If from_mag isn't specified the program will cycle through provided magnitudes and choose one. Note that all
         magnitudes are first converted to V, and then to the requested magnitude.

        :param to_mag: magnitude to convert to
        :param from_mag: magnitude to convert from
        :return:
        """
        allowed_mags = "UBVJIHKLMN"

        if from_mag:
            if to_mag == 'V':  # If V mag is requested (1/3) - from mag specified
                return self._convert_to_from('V', from_mag)
            if from_mag == 'V':
                magV = self.magV
            else:
                magV = self._convert_to_from('V', from_mag)

            return self._convert_to_from(to_mag, 'V', magV)

        # if we can convert from any magnitude, try V first
        elif not isNanOrNone(self.magV):
            if to_mag == 'V':  # If V mag is requested (2/3) - no need to convert
                return self.magV
            else:
                return self._convert_to_from(to_mag, 'V', self.magV)
        else:  # Otherwise lets try all other magnitudes in turn
            order = "UBJHKLMN"  # V is the intermediate step from the others, done by default if possible
            for mag_letter in order:
                try:
                    magV = self._convert_to_from('V', mag_letter)
                    if to_mag == 'V':  # If V mag is requested (3/3) - try all other mags to convert
                        logging.debug('Converted to magV from {0} got {1}'.format(mag_letter, magV))
                        return magV
                    else:
                        mag_val = self._convert_to_from(to_mag, 'V', magV)
                        logging.debug('Converted to mag{0} from {1} got {2}'.format(to_mag, mag_letter, mag_val))
                        return mag_val
                except ValueError:
                    continue  # this conversion may not be possible, try another

            raise ValueError('Could not convert from any provided magnitudes')