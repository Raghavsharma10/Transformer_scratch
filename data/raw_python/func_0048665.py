def sorteditems(self, multi=False):
        # type: (bool) -> Iterator[Tuple[Hashable, Any]]
        """
        Return an iterator of ``(key, value)`` pairs, sorted by key.

        :param multi: If set to `True` the iterator returned will have a pair
                      for each value of each key.  Otherwise it will only
                      contain pairs for the lasted added of each key.

        """
        for key in sorted(dict.keys(self)):
            if multi:
                for value in self.getlist(key):
                    yield key, value
            else:
                yield key, self[key]