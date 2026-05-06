def from_array(cls, array):
        """Returns a |Timegrid| instance based on two date and one period
        information stored in the first 13 rows of a |numpy.ndarray| object.
        """
        try:
            return cls(Date.from_array(array[:6]),
                       Date.from_array(array[6:12]),
                       Period.fromseconds(array[12]))
        except IndexError:
            raise IndexError(
                f'To define a Timegrid instance via an array, 13 '
                f'numbers are required.  However, the given array '
                f'consist of {len(array)} entries/rows only.')