def delete(self, key):
        """Remove a value from the `DotDict`.

        The `key` parameter can either be a regular string key,
        e.g. "foo", or it can be a string key with dot notation,
        e.g. "foo.bar.baz", to signify a nested element.

        If the key does not exist in the `DotDict`, it will continue
        silently.

        Args:
            key (str): The key to remove.
        """
        dct = self
        keys = key.split('.')
        last_key = keys[-1]
        for k in keys:
            # if the key is the last one, e.g. 'z' in 'x.y.z', try
            # to delete it from its dict.
            if k == last_key:
                del dct[k]
                break

            # if the dct is a DotDict, get the value for the key `k` from it.
            if isinstance(dct, DotDict):
                dct = super(DotDict, dct).__getitem__(k)

            # otherwise, just get the value from the default __getitem__
            # implementation.
            else:
                dct = dct.__getitem__(k)
                if not isinstance(dct, (DotDict, dict)):
                    raise KeyError(
                        'Subkey "{}" in "{}" invalid for deletion'.format(k, key)
                    )