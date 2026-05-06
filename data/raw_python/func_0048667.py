def to_dict(self, flat=True):
        """
        Return the contents as regular dict.  If `flat` is `True` the
        returned dict will only have the first item present, if `flat` is
        `False` all values will be returned as lists.

        :param flat: If set to `False` the dict returned will have lists
                     with all the values in it.  Otherwise it will only
                     contain the last value for each key.
        :return: a :class:`dict`

        """
        if flat:
            return dict(self.items())
        return dict(self.lists())