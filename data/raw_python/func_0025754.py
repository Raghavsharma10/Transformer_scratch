def handle(self, data):
        """
        Writes each sample to a csv file.
        :param data: the samples.
        :return:
        """
        self.logger.debug("Handling " + str(len(data)) + " data items")
        for datum in data:
            if isinstance(datum, dict):
                # these have to wrapped in a list for python 3.4 due to a change in the implementation
                # of OrderedDict in python 3.5+ (which means .keys() and .values() are sequences in 3.5+)
                if self._first:
                    self._csv.writerow(list(datum.keys()))
                    self._first = False
                self._csv.writerow(list(datum.values()))
            elif isinstance(datum, list):
                self._csv.writerow(datum)
            else:
                self.logger.warning("Ignoring unsupported data type " + str(type(datum)) + " : " + str(datum))