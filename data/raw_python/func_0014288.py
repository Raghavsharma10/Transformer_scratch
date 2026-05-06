def from_csv(cls, path):
        """
        Get box vectors from comma-separated values in file `path`.

        The csv file must containt only one line, which in turn can contain
        three values (orthogonal vectors) or nine values (triclinic box).

        The values should be in nanometers.

        Parameters
        ----------
        path : str
            Path to CSV file

        Returns
        -------
        vectors : simtk.unit.Quantity([3, 3], unit=nanometers
        """
        with open(path) as f:
            fields = map(float, next(f).split(','))
        if len(fields) == 3:
            return u.Quantity([[fields[0], 0, 0],
                               [0, fields[1], 0],
                               [0, 0, fields[2]]], unit=u.nanometers)
        elif len(fields) == 9:
            return u.Quantity([fields[0:3],
                               fields[3:6],
                               fields[6:9]], unit=u.nanometers)
        else:
            raise ValueError('This type of CSV is not supported. Please '
                             'provide a comma-separated list of three or nine '
                             'floats in a single-line file.')