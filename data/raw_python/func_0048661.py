def getlist(self, key, type_=None):
        # type: (Hashable, Callable) -> List[Any]
        """
        Return the list of items for a given key. If that key is not in the
        `MultiDict`, the return value will be an empty list.  Just as `get`
        `getlist` accepts a `type` parameter.  All items will be converted
        with the callable defined there.

        :param key: The key to be looked up.
        :param type_: A callable that is used to cast the value in the
                     :class:`MultiDict`.  If a :exc:`ValueError` is raised
                     by this callable the value will be removed from the list.
        :return: a :class:`list` of all the values for the key.

        """
        try:
            rv = dict.__getitem__(self, key)
        except KeyError:
            return []
        if type_ is None:
            return list(rv)
        result = []
        for item in rv:
            try:
                result.append(type_(item))
            except ValueError:
                pass
        return result