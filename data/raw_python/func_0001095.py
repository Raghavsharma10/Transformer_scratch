def insert(self, var, value, index=None):
        """Insert at the index.

        If the index is not provided appends to the end of the list.
        """
        current = self.__get(var)
        if not isinstance(current, list):
            raise KeyError("%s: is not a list" % var)
        if index is None:
            current.append(value)
        else:
            current.insert(index, value)
        if self.auto_save:
            self.save()