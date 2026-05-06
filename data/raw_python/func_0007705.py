def _read_as_dict(self):
        """
        Read the data contained in all entries as a list of
        dictionaries with the headers as the dictionary keys

        :return: list of dicts containing all tabular data
        """
        data = list()
        for row in self._rows:
            row_data = OrderedDict()
            for i, header in enumerate(self.headers):
                row_data[header.cget('text')] = row[i].get()

            data.append(row_data)

        return data