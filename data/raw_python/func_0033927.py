def get(self, index):
        """Gets data values for specified :index:.

        :index: Index for which to get data.
        :returns: A list in form
        [parent, name, priority, comment, done, children].

        """
        data = self.data
        index2 = self._split(index)
        for c in index2[:-1]:
            i = int(c) - 1
            data = data[i][4]
        return [index[:-2] or ""] + data[int(index[-1]) - 1]