def to_array(self, variables):
        """
        Converts the clamping to a 1-D array with respect to the given variables

        Parameters
        ----------
        variables : list[str]
            List of variables names


        Returns
        -------
        `numpy.ndarray`_
            1-D array where position `i` correspond to the sign of the clamped variable at
            position `i` in the given list of variables


        .. _numpy.ndarray: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html#numpy.ndarray
        """
        arr = np.zeros(len(variables), np.int8)
        dc = dict(self)

        for i, var in enumerate(variables):
            arr[i] = dc.get(var, arr[i])

        return arr