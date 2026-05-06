def get_singleton(self):
        """If the row only has one column, return that value; otherwise raise.

        Raises:
            ValueError, if count of columns is not 1.
        """
        only_value = None
        for value in six.itervalues(self.ordered_dict):
            # This loop will raise if it runs more than once.
            if only_value is not None:
                raise ValueError("%r is not a singleton." % self)

            only_value = value

        if only_value is self.__UnsetSentinel or only_value is None:
            raise ValueError("%r is empty." % self)

        return only_value