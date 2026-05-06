def load(self, filename=None, refresh=False):
        """
        Try to load the data from a pre existing data file if it exists.
        If the data file does not exist, refresh the data and save it in
        the data file for future use.
        The data file is a json file.

        :param filename: The filename to save or fetch the data from.
        :param refresh:  Whether to force refresh the data or not
        """
        filename = filename or self.data_file()
        dirname = os.path.dirname(filename)

        if refresh is False:
            try:
                data = None
                with open(filename) as fp:
                    data = json.load(fp)
                self.clear()
                self.update(data)
                return
            except (ValueError, IOError):
                # Refresh data if reading gave errors
                pass

        data = self.refresh()
        self.clear()
        self.update(data)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(filename, 'w') as fp:
            json.dump(data, fp,
                      sort_keys=True,
                      indent=2,
                      separators=(',', ': '))