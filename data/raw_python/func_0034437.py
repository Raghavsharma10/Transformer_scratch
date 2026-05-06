def append(self, val):
        """Appends the object to the end of the values list.  Will also set the value to the first
        item in the values list

        :param val: Object to append
        :type val: primitive
        """
        self.values.append(val)
        self.value = self.values[0]