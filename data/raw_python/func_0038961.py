def labels(self, values):
        """Class labels (such as 1, 2, -1, 'A', 'B' etc.) for each sample in the dataset."""
        if isinstance(values, dict):
            if self.__data is not None and len(self.__data) != len(values):
                raise ValueError(
                    'number of samples do not match the previously assigned data')
            elif set(self.keys) != set(list(values)):
                raise ValueError('sample ids do not match the previously assigned ids.')
            else:
                self.__labels = values
        else:
            raise ValueError('labels input must be a dictionary!')