def to_dict(self):
        """Transform to dictionary

        Returns:
            dict: dictionary with same content
        """
        return {key: self.__getitem__(key).value for key in self.options()}