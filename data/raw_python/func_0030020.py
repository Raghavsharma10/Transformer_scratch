def python_cast(self, v):
        """Cast a value to the type of the column.

        Primarily used to check that a value is valid; it will throw an
        exception otherwise

        """

        if self.type_is_time():
            dt = dateutil.parser.parse(v)

            if self.datatype == Column.DATATYPE_TIME:
                dt = dt.time()
            if not isinstance(dt, self.python_type):
                raise TypeError('{} was parsed to {}, expected {}'.format(v, type(dt), self.python_type))

            return dt
        else:
            # This isn't calling the python_type method -- it's getting a python type, then instantialting it,
            # such as "int(v)"
            return self.python_type(v)