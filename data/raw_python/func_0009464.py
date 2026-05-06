def addNoise(self, nlf_coeff, date=None, info='', error=None):
        '''
        Args:
            nlf_coeff (list)
            error (float): absolute
            info (str): additional information
            date (str): "DD Mon YY" e.g. "30 Nov 16"
        '''
        date = _toDate(date)
        d = self.coeffs['noise']
        d.insert(_insertDateIndex(date, d), [date, info, nlf_coeff, error])