def classes(self, values):
        """Classes setter."""
        if isinstance(values, dict):
            if self.__data is not None and len(self.__data) != len(values):
                raise ValueError(
                    'number of samples do not match the previously assigned data')
            elif set(self.keys) != set(list(values)):
                raise ValueError('sample ids do not match the previously assigned ids.')
            else:
                self.__classes = values
        else:
            raise ValueError('classes input must be a dictionary!')