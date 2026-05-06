def append(self, data, itemsize=None):
        """
        Append data to the end.

        Parameters
        ----------

        data : array_like
            An array, any object exposing the array interface, an object
            whose __array__ method returns an array, or any (nested) sequence.

        itemsize:  int or 1-D array
            If `itemsize is an integer, N, the array will be divided
            into elements of size N. If such partition is not possible,
            an error is raised.

            If `itemsize` is 1-D array, the array will be divided into
            elements whose succesive sizes will be picked from itemsize.
            If the sum of itemsize values is different from array size,
            an error is raised.
        """

        self.insert(len(self), data, itemsize)