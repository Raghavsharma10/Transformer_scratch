def get(self):
        """
        Retrieve the GUI elements for program use.

        :return: a dictionary containing all \
        of the data from the key/value entries
        """
        data = dict()
        for label, entry in zip(self.keys, self.values):
            data[label.cget('text')] = entry.get()

        return data