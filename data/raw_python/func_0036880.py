def parse(cls, filename, max_life=None):
        """ Parse barcode from gudhi output. """
        data = np.genfromtxt(filename)
        #data = np.genfromtxt(filename, dtype= (int, int, float, float))

        if max_life is not None:
            data[np.isinf(data)] = max_life

        return data