def values(self, multi=False):
        # type: (bool) -> Iterator[Any]
        """
        Yield the last value on every key list.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the lasted added of each key.

        """
        for values in itervalues(self):
            if multi:
                for value in values:
                    yield value
            else:
                yield values[-1]