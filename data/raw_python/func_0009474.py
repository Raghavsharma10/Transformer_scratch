def getCoeff(self, name, light=None, date=None):
        '''
        try to get calibration for right light source, but
        use another if they is none existent
        '''
        d = self.coeffs[name]

        try:
            c = d[light]
        except KeyError:
            try:
                k, i = next(iter(d.items()))
                if light is not None:
                    print(
                        'no calibration found for [%s] - using [%s] instead' % (light, k))
            except StopIteration:
                return None
            c = i
        except TypeError:
            # coeff not dependent on light source
            c = d
        return _getFromDate(c, date)