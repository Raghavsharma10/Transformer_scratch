def add_field(self, name, data):
        """
        Add a new field to the datamat.

        Parameters:
            name : string
                Name of the new field
            data : list
                Data for the new field, must be same length as all other fields.
        """
        if name in self._fields:
            raise ValueError
        if not len(data) == self._num_fix:
            raise ValueError
        self._fields.append(name)
        self.__dict__[name] = data