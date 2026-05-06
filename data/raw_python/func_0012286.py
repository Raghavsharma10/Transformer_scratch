def to_dict(self):
        """Transform to dictionary

        Returns:
            dict: dictionary with same content
        """
        return {sect: self.__getitem__(sect).to_dict()
                for sect in self.sections()}