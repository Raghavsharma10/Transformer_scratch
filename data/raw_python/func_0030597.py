def parse(v):
        """
        Parse a slice string, of the same form as used by __getitem__

        >>> Slice.parse("2:3,7,10:12")

        :param v: Input string
        :return: A list of tuples, one for each element of the slice string
        """

        parts = v.split(',')

        slices = []

        for part in parts:
            p = part.split(':')

            if len(p) == 1:
                slices.append(int(p[0]))
            elif len(p) == 2:
                slices.append(tuple(p))
            else:
                raise ValueError("Too many ':': {}".format(part))

        return slices