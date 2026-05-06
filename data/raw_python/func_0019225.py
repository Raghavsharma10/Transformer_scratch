def sum_coefs(self):
        """The sum of all AR and MA coefficients"""
        return numpy.sum(self.ar_coefs) + numpy.sum(self.ma_coefs)