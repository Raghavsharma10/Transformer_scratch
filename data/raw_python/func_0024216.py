def get_all(self):
        # type: () -> typing.Iterable[typing.Tuple[str, str]]
        """Returns an iterable of all (name, value) pairs.

        If a header has multiple values, multiple pairs will be
        returned with the same name.
        """
        for name, values in six.iteritems(self._as_list):
            for value in values:
                yield (name, value)