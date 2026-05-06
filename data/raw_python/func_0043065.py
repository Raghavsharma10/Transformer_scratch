def get(self, key, default=None):
        """Get a value from the `DotDict`.

        The `key` parameter can either be a regular string key,
        e.g. "foo", or it can be a string key with dot notation,
        e.g. "foo.bar.baz", to signify a nested lookup.

        The default value is returned if any level of the key's
        components are not found.

        Args:
            key (str): The key to get the value for.
            default: The return value should the given key
                not exist in the `DotDict`.
        """
        # if there are no dots in the key, its a normal get
        if key.count('.') == 0:
            return super(DotDict, self).get(key, default)

        # set the return value to the default
        value = default

        # split the key into the first component and the rest of
        # the components. the first component corresponds to this
        # DotDict. the remainder components correspond to any nested
        # DotDicts.
        first, remainder = key.split('.', 1)
        if first in self:
            value = super(DotDict, self).get(first, default)

            # if the value for the key at this level is a dictionary,
            # then pass the remainder to that DotDict.
            if isinstance(value, (dict, DotDict)):
                return DotDict(value).get(remainder, default)

            # TODO: support lists

        return value