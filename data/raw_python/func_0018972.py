def add(self, name: Any) -> None:
        """Before adding a new name, it is checked to be valid variable
        identifiers.

        >>> from hydpy.core.devicetools import Keywords
        >>> keywords = Keywords('first_keyword', 'second_keyword',
        ...                     'keyword_3', 'keyword_4',
        ...                     'keyboard')
        >>> keywords.add('1_test')   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: While trying to add the keyword `1_test` to device ?, \
the following error occurred: The given name string `1_test` does not \
define a valid variable identifier.  ...

        >>> keywords
        Keywords("first_keyword", "keyboard", "keyword_3", "keyword_4",
                 "second_keyword")

        After correcting the string, everything works fine:

        >>> keywords.add('one_test')
        >>> keywords
        Keywords("first_keyword", "keyboard", "keyword_3", "keyword_4",
                 "one_test", "second_keyword")
        """
        self._check_keywords([str(name)])
        super().add(str(name))