def tuple_len(self):
        """
        Length of tuples produced by this generator.
        """
        try:
            return self._tuple_len
        except AttributeError:
            raise NotImplementedError("Class {} does not implement attribute 'tuple_len'.".format(self.__class__.__name__))