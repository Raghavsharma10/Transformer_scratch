def _npdelta(self, a, delta):
        """Numpy: Modifying Array Values
            http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html"""
        for x in np.nditer(a, op_flags=["readwrite"]):
            delta += x
            x[...] = delta
        return a