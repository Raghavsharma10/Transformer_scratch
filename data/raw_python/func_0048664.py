def items(self, multi=False):
        # type: (bool) -> Iterator[Tuple[Hashable, Any]]
        """
        Return an iterator of ``(key, value)`` pairs.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the lasted added of each key.
        """
        for key, values in iteritems(self):
            if multi:
                for value in values:
                    yield key, value
            else:
                yield key, values[-1]