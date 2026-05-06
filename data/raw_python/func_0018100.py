def exportable_keys(self):
    """Return a list of keys that are exportable from this tuple.

    Returns all keys that are not private in any of the tuples.
    """
    keys = collections.defaultdict(list)
    for tup in self._tuples:
      for key, private in tup._keys_and_privacy().items():
        keys[key].append(private)
    return [k for k, ps in keys.items() if not any(ps)]