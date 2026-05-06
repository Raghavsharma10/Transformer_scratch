def pop(self, key, default=NotDefined):
        # type: (Hashable, Any) -> Any
        """
        Pop the last item for a list on the dict.  Afterwards the
        key is removed from the dict, so additional values are discarded:
        >>> d = MultiValueDict({"foo": [1, 2, 3]})
        >>> d.pop("foo")
        1
        >>> "foo" in d
        False

        :param key: the key to pop.
        :param default: if provided the value to return if the key was
                        not in the dictionary.
        """
        try:
            return dict.pop(self, key)[-1]
        except LookupError:
            if default is NotDefined:
                raise MultiValueDictKeyError(key)
            return default