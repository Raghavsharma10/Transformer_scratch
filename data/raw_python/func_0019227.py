def get_a(values, n):
        """Extract the independent variables of the given values and return
        them as a matrix with n columns in a form suitable for the least
        squares approach applied in method |ARMA.update_ar_coefs|.
        """
        m = len(values)-n
        a = numpy.empty((m, n), dtype=float)
        for i in range(m):
            i0 = i-1 if i > 0 else None
            i1 = i+n-1
            a[i] = values[i1:i0:-1]
        return numpy.array(a)