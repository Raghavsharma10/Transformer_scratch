def response(self):
        """Return the response to a standard dt impulse."""
        values = []
        sum_values = 0.
        ma_coefs = self.ma_coefs
        ar_coefs = self.ar_coefs
        ma_order = self.ma_order
        for idx in range(len(self.ma.delays)):
            value = 0.
            if idx < ma_order:
                value += ma_coefs[idx]
            for jdx, ar_coef in enumerate(ar_coefs):
                zdx = idx-jdx-1
                if zdx >= 0:
                    value += ar_coef*values[zdx]
            values.append(value)
            sum_values += value
        return numpy.array(values)