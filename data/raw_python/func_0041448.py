def get_source(self, key, name_spaces=None, default_prefix=''):
        """Generates the dictionary key for the serialized representation
        based on the instance variable source and a provided key.

        :param str key: name of the field in model
        :returns: self.source or key
        """
        source = self.source or key
        prefix = default_prefix
        if name_spaces and self.name_space and self.name_space in name_spaces:
            prefix = ''.join([name_spaces[self.name_space], ':'])
        return ''.join([prefix, source])