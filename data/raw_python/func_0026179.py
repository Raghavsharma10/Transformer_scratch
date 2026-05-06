def as_bool(self, key):
        """
        Accepts a key as input. The corresponding value must be a string or
        the objects (``True`` or 1) or (``False`` or 0). We allow 0 and 1 to
        retain compatibility with Python 2.2.

        If the string is one of  ``True``, ``On``, ``Yes``, or ``1`` it returns
        ``True``.

        If the string is one of  ``False``, ``Off``, ``No``, or ``0`` it returns
        ``False``.

        ``as_bool`` is not case sensitive.

        Any other input will raise a ``ValueError``.

        >>> a = ConfigObj()
        >>> a['a'] = 'fish'
        >>> a.as_bool('a')
        Traceback (most recent call last):
        ValueError: Value "fish" is neither True nor False
        >>> a['b'] = 'True'
        >>> a.as_bool('b')
        1
        >>> a['b'] = 'off'
        >>> a.as_bool('b')
        0
        """
        val = self[key]
        if val == True:
            return True
        elif val == False:
            return False
        else:
            try:
                if not isinstance(val, string_types):
                    # TODO: Why do we raise a KeyError here?
                    raise KeyError()
                else:
                    return self.main._bools[val.lower()]
            except KeyError:
                raise ValueError('Value "%s" is neither True nor False' % val)