def convert_numpy_type(cls, dtype):
        """Convert a numpy dtype into a Column datatype. Only handles common
        types.

        Implemented as a function to decouple from numpy

        """

        m = {
            'int64': cls.DATATYPE_INTEGER64,
            'float64': cls.DATATYPE_FLOAT,
            'object': cls.DATATYPE_TEXT  # Hack. Pandas makes strings into object.
        }

        t = m.get(dtype.name, None)

        if not t:
            raise TypeError(
                "Failed to convert numpy type: '{}' ".format(
                    dtype.name))

        return t