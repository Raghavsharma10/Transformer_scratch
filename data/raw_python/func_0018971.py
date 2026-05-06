def update(self, *names: Any) -> None:
        """Before updating, the given names are checked to be valid
        variable identifiers.

        >>> from hydpy.core.devicetools import Keywords
        >>> keywords = Keywords('first_keyword', 'second_keyword',
        ...                     'keyword_3', 'keyword_4',
        ...                     'keyboard')
        >>> keywords.update('test_1', 'test 2')   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: While trying to add the keyword `test 2` to device ?, \
the following error occurred: The given name string `test 2` does not \
define a valid variable identifier.  ...

        Note that even the first string (`test1`) is not added due to the
        second one (`test 2`) being invalid.

        >>> keywords
        Keywords("first_keyword", "keyboard", "keyword_3", "keyword_4",
                 "second_keyword")

        After correcting the second string, everything works fine:

        >>> keywords.update('test_1', 'test_2')
        >>> keywords
        Keywords("first_keyword", "keyboard", "keyword_3", "keyword_4",
                 "second_keyword", "test_1", "test_2")
        """
        _names = [str(name) for name in names]
        self._check_keywords(_names)
        super().update(_names)